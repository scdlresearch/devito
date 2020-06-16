import numpy as np # noqa

from examples.seismic import demo_model, setup_geometry, seismic_args
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              space_order=4, nbl=10, preset='layers-tti', **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing,
                       space_order=space_order, nbl=nbl, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbl=10,
        kernel='centered', **kwargs):

    solver = tti_setup(shape, spacing, tn, space_order, nbl, **kwargs)

    rec, u, v, summary = solver.forward(autotune=autotune, kernel=kernel)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = seismic_args(description)
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    args = parser.parse_args()

    # Switch to TTI kernel if input is acoustic kernel
    kernel = 'centered' if args.kernel in ['OT2', 'OT4'] else args.kernel
    preset = 'layers-tti-noazimuth' if args.azi else 'layers-tti'

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)
    dtype = eval((''.join(['np.', args.dtype])))

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune,
        opt=args.opt, kernel=kernel, preset=preset, dtype=dtype)
