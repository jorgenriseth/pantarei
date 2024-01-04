import os
import shutil
from pathlib import Path
import dolfin
import numpy as np

import pantarei as pr
from pantarei import geo2mesh, mesh2xdmf, xdmf2hdf, geo2hdf, clean_tmp, hdf2fenics


def get_test_directories():
    parent = Path(__file__).parent
    outdir = parent / "outdir"
    outdir.mkdir(exist_ok=True)
    return parent, outdir


def test_geo2mesh():
    parent, outdir = get_test_directories()

    geo2mesh(infile=parent/"square.geo", outfile=outdir/"square.mesh", dim=2)
    assert "square.mesh" in  map(lambda x: str(x.name), outdir.iterdir())

    # clean_tmp(outdir, "mesh")


def test_mesh2xdmf():
    parent, outdir = get_test_directories()
    mesh2xdmf(parent / "square.mesh", outdir, dim=2)

    assert "mesh.xdmf" in os.listdir(outdir)
    assert "boundaries.xdmf" in os.listdir(outdir)
    assert "subdomains.xdmf" in os.listdir(outdir)

    # Delete output folder
    # clean_tmp(outdir, "hdf")
    # clean_tmp(outdir, "xdmf")


def test_xdmf2hdf():
    parent, outdir = get_test_directories()
    xdmf2hdf(outdir, outdir / "square.hdf")
    assert "square.hdf" in os.listdir(outdir)
    # clean_tmp(outdir, "hdf")


def test_geo2hdf():
    parent, outdir = get_test_directories()
    Path(parent / "tmp").mkdir(exist_ok=True)
    geo2hdf(parent / "square.geo", outdir / "square.hdf", dim=2)
    assert "square.hdf" in os.listdir(outdir)
    # assert len(os.listdir(outdir)) == 1
    # clean_tmp(outdir, "hdf")


def test_hdf2fenics():
    parent, _ = get_test_directories()
    mesh, subdomains, boundaries = hdf2fenics(parent / "square.hdf")

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
    test_geo2hdf()
    outdir = Path(__file__).parent / "outdir"
    if outdir.exists():
        shutil.rmtree(outdir)
