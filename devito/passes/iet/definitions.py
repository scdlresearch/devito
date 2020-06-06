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
from devito.tools import as_tuple

__all__ = ['DataManager', 'Storage']


Definition = namedtuple('Definition', 'decl alloc free region')
Definition.__new__.__defaults__ = (None,) * len(Definition._fields)

class DefinitionMapper(OrderedDict):

    def __setitem__(self, k, v):
        if not v or isinstance(v, Definition):
            v = [v]
        elif not isinstance(v, Iterable):
            raise ValueError
        super(DefinitionMapper, self).__setitem__(k, v)


class Storage(OrderedDict):

    def make_site(self, site):
        return self.setdefault(site, DefinitionMapper())


class DataManager(object):

    _Parallelizer = Ompizer

    def _alloc_object_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate a LocalObject in the low latency memory.
        """
        mapper = storage.make_site(site)

        mapper[obj] = Definition(c.Value(obj._C_typename, obj.name))

    def _alloc_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array in the low latency memory.
        """
        mapper = storage.make_site(site)

        if obj in mapper:
            return

        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = "__attribute__((aligned(%d)))" % obj._data_alignment
        value = "%s%s %s" % (obj.name, shape, alignment)

        mapper[obj] = Definition(c.POD(obj.dtype, value))

    def _alloc_scalar_on_low_lat_mem(self, site, expr, storage):
        """
        Allocate a Scalar in the low latency memory.
        """
        obj = expr.write
        mapper = storage.make_site(site)

        if obj in mapper:
            return

        mapper[obj] = None  # Placeholder to avoid reallocation
        mapper[expr] = Definition(LocalExpression(**expr.args))

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        """Allocate an Array in the high bandwidth memory."""
        mapper = storage.make_site(site)

        if obj in mapper:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        mapper[obj] = Definition(decl, alloc, free)

    def _alloc_array_slice_per_thread(self, site, obj, storage):
        """
        For an Array whose outermost is a ThreadDimension, allocate each of its slices
        in the high bandwidth memory.
        """
        mapper = storage.make_site(site)

        if obj in mapper:
            return

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

        mapper[obj] = [Definition(decl, alloc, free)]

        # Construct parallel pointer allocation
        shape = "".join("[%s]" % i for i in obj.symbolic_shape[1:])
        alloc = "posix_memalign((void**)&%s[%s], %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, tid.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s[%s])' % (obj.name, tid.name))

        mapper[obj].append(Definition(None, alloc, free, tid))

    def _dump_storage(self, iet, storage):
        subs = {}
        site_mapper = {}
        for site, mapper in storage.items():
            for obj, definitions in mapper.items():
                for definition in as_tuple(definitions):
                    if not definition:
                        continue

                    try:
                        if obj.is_Expression:
                            # Expr -> LocalExpr
                            subs[obj] = definition.decl
                            continue
                    except AttributeError:
                        pass

                    site_mapper[site]
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

        processed = Transformer(subs, nested=True).visit(iet)

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
