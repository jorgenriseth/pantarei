from pantarei.meshprocessing import geo2mesh, mesh2xdmf, xdmf2hdf, geo2hdf, clean_tmp, hdf2fenics
from pantarei.domain import unpack_domain
import os
from pathlib import Path
import dolfin
import numpy as np


def get_test_directories():
    parent = Path(__file__).parent.resolve()
    outdir = parent / "outdir"
    outdir.mkdir(exist_ok=True)
    return parent, outdir


def test_geo2mesh():
    parent, outdir = get_test_directories()

    geo2mesh(infile=parent/"square.geo", outfile=outdir/"square.mesh", dim=2)
    assert "square.mesh" in os.listdir(outdir)

    clean_tmp(outdir, "mesh")


def test_mesh2xdmf():
    parent, outdir = get_test_directories()
    mesh2xdmf(parent / "square.mesh", outdir, dim=2)

    assert "mesh.xdmf" in os.listdir(outdir)
    assert "boundaries.xdmf" in os.listdir(outdir)
    assert "subdomains.xdmf" in os.listdir(outdir)

    # Delete output folder
    clean_tmp(outdir, "h5")
    clean_tmp(outdir, "xdmf")


def test_xdmf2hdf():
    parent, outdir = get_test_directories()
    xdmf2hdf(parent / "xdmfdir", outdir / "square.h5")
    assert "square.h5" in os.listdir(outdir)
    clean_tmp(outdir, "h5")


def test_geo2h5():
    parent, outdir = get_test_directories()
    Path(parent / "tmp").mkdir(exist_ok=True)
    geo2hdf(parent / "square.geo", outdir / "square.h5")
    assert "square.h5" in os.listdir(outdir)
    assert len(os.listdir(outdir)) == 1
    clean_tmp(outdir, "h5")


def test_h52fenics():
    parent, _ = get_test_directories()
    mesh, subdomains, boundaries = unpack_domain(hdf2fenics(parent / "square.h5"))

    for cell in dolfin.cpp.mesh.cells(mesh):
        assert subdomains[cell] == 1

    for edge in dolfin.cpp.mesh.edges(mesh):
        if boundaries[edge] == 1:
            assert np.allclose(edge.midpoint().array(),
                               np.array([0.5, 0., 0.]))
        elif boundaries[edge] == 2:
            assert np.allclose(edge.midpoint().array(),
                               np.array([1., 0.5, 0.]))
        elif boundaries[edge] == 3:
            assert np.allclose(edge.midpoint().array(),
                               np.array([0.5, 1., 0.]))
        elif boundaries[edge] == 4:
            assert np.allclose(edge.midpoint().array(),
                               np.array([0., 0.5, 0.]))
        else:
            assert boundaries[edge] == 0


if __name__ == '__main__':
    test_geo2mesh()
    test_mesh2xdmf()
    test_xdmf2hdf()
    test_geo2h5()
