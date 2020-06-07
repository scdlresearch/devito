"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import Iterable, OrderedDict, namedtuple

import cgen as c

from devito.ir import (ArrayCast, Element, List, LocalExpression, FindSymbols,
                       MapExprStmts, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.openmp import Ompizer
from devito.symbolics import ccode
from devito.tools import as_mapper

__all__ = ['DataManager', 'Storage']


MetaSite = namedtuple('Definition', 'decls allocs frees pallocs pfrees')


class Storage(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(Storage, self).__init__(*args, **kwargs)
        self.defined = set()

    def update(self, obj, site, **kwargs):
        if obj in self.defined:
            return

        try:
            metasite = self[site]
        except KeyError:
            metasite = self.setdefault(site, MetaSite([], [], [], [], []))

        for k, v in kwargs.items():
            getattr(metasite, k).append(v)

        self.defined.add(obj)

    def map(self, obj, k, v):
        if obj in self.defined:
            return

        self[k] = v
        self.defined.add(obj)

class DataManager(object):

    _Parallelizer = Ompizer

    def _alloc_object_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate a LocalObject in the low latency memory.
        """
        storage.update(obj, site, decls=c.Value(obj._C_typename, obj.name))

    def _alloc_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array in the low latency memory.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = "__attribute__((aligned(%d)))" % obj._data_alignment
        value = "%s%s %s" % (obj.name, shape, alignment)

        storage.update(obj, site, decls=c.POD(obj.dtype, value))

    def _alloc_scalar_on_low_lat_mem(self, site, expr, storage):
        """
        Allocate a Scalar in the low latency memory.
        """
        storage.map(expr.write, expr, LocalExpression(**expr.args))

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate an Array in the high bandwidth memory.
        """
        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        storage.update(obj, site, decls=decl, allocs=alloc, frees=free)

    def _alloc_array_slice_per_thread(self, site, obj, storage):
        """
        For an Array whose outermost is a ThreadDimension, allocate each of its slices
        in the high bandwidth memory.
        """
        # Construct the definition for a pointer array that is `nthreads` long
        tid = obj.dimensions[0]
        assert tid.is_Thread

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "[%s]" % tid.symbolic_size
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        # Construct parallel pointer allocation
        shape = "".join("[%s]" % i for i in obj.symbolic_shape[1:])
        palloc = "posix_memalign((void**)&%s[%s], %d, sizeof(%s%s))"
        palloc = palloc % (obj.name, tid.name, obj._data_alignment, obj._C_typedata,
                           shape)
        palloc = c.Statement(palloc)

        pfree = c.Statement('free(%s[%s])' % (obj.name, tid.name))

        storage.update(obj, site, decls=decl, allocs=alloc, frees=free,
                       pallocs=(tid, palloc), pfrees=(tid, pfree))

    def _dump_storage(self, iet, storage):
        mapper = {}
        for k, v in storage.items():
            # Expr -> LocalExpr ?
            if k.is_Expression:
                mapper[k] = v
                continue

            from IPython import embed; embed()


#        # Introduce symbol definitions going into the high bandwidth memory
#        for scope, handle in storage._high_bw_mem.items():
#            header = []
#            footer = []
#            for decl, alloc, free in handle.values():
#                if decl is None:
#                    header.append(alloc)
#                else:
#                    header.extend([decl, alloc])
#                footer.append(free)
#            if header or footer:
#                header.append(c.Line())
#                footer.insert(0, c.Line())
#                body = List(header=header,
#                            body=as_tuple(mapper.get(scope)) + scope.body,
#                            footer=footer)
#                mapper[scope] = scope._rebuild(body=body, **scope.args_frozen)
#
#        # Introduce symbol definitions for thread-shared arrays
#        for (scope, tid), handle in storage._high_bw_mem_threaded.items():
#            body = List(header=[alloc for alloc, _ in handle.values()])
#            top = self._Parallelizer._Region(body, tid.symbolic_size)
#
#            body = List(header=[free for _, free in handle.values()])
#           bottom = self._Parallelizer._Region(body, tid.symbolic_size)
#
#            a = List(body=[top, mapper.get(scope, scope), bottom])
#            mapper[scope] = a
#            from IPython import embed; embed()
#

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed

    @iet_pass
    def place_definitions(self, iet, **kwargs):
        """
        Create a new IET with symbols allocated/deallocated in some memory space.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        storage = Storage()

        already_defined = list(iet.parameters)

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression:
                if k.is_definition:
                    site = v[-1] if v else iet
                    self._alloc_scalar_on_low_lat_mem(site, k, storage)
                    continue
                objs = [k.write]
            elif k.is_Dereference:
                already_defined.append(k.array0)
                objs = [k.array1]
            elif k.is_Call:
                objs = k.arguments

            for i in objs:
                try:
                    if i.is_LocalObject:
                        site = v[-1] if v else iet
                        self._alloc_object_on_low_lat_mem(site, i, storage)
                    elif i.is_Array:
                        if i in already_defined:
                            # The Array is passed as a Callable argument
                            continue

                        site = iet
                        if i._mem_local:
                            # If inside a ParallelRegion, make sure we allocate
                            # inside of it
                            for n in v:
                                if n.is_ParallelBlock:
                                    site = n
                                    break
                            if i._mem_heap:
                                self._alloc_array_on_high_bw_mem(site, i, storage)
                            else:
                                self._alloc_array_on_low_lat_mem(site, i, storage)
                        else:
                            if i._mem_heap:
                                if i.dimensions[0].is_Thread:
                                    # Optimization: each thread allocates its own
                                    # logically private slice
                                    self._alloc_array_slice_per_thread(site, i, storage)
                                else:
                                    self._alloc_array_on_high_bw_mem(site, i, storage)
                            else:
                                self._alloc_array_on_low_lat_mem(site, i, storage)
                except AttributeError:
                    # E.g., a generic SymPy expression
                    pass

        iet = self._dump_storage(iet, storage)

        return iet, {}

    @iet_pass
    def place_casts(self, iet):
        """
        Create a new IET with the necessary type casts.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        functions = FindSymbols().visit(iet)
        need_cast = {i for i in functions if i.is_Tensor}

        # Make the generated code less verbose by avoiding unnecessary casts
        indexed_names = {i.name for i in FindSymbols('indexeds').visit(iet)}
        need_cast = {i for i in need_cast if i.name in indexed_names or i.is_Array}

        casts = tuple(ArrayCast(i) for i in iet.parameters if i in need_cast)
        iet = iet._rebuild(body=casts + iet.body)

        return iet, {}
