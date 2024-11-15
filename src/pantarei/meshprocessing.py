#!/usr/bin/env python
import glob
import os
import uuid
from pathlib import Path
from typing import Optional, TypeAlias, Union

import meshio
import SVMTK as svmtk
from dolfin import HDF5File, Mesh, MeshFunction, MeshValueCollection, XDMFFile
from loguru import logger

from pantarei.domain import Domain


StrPath: TypeAlias = Union[str, Path]


def create_tmpdir(tmpdir: Optional[StrPath]) -> Path:
    if tmpdir is None:
        tmppath = Path(f"/tmp/pantarei/{uuid.uuid4()}")
    else:
        tmppath = Path(tmpdir)
    tmppath.mkdir(exist_ok=True, parents=True)
    return tmppath


def prune_z_0(mesh):
    return mesh.points[:, :2]


def clean_tmp(directory, suffix, no_output=False):
    """Delete the given directory, with content."""
    dirpath = Path(directory)
    for f in glob.glob(f"{dirpath}/*{suffix}"):
        os.remove(f)

    if len(os.listdir(dirpath)) == 0:
        os.rmdir(dirpath)
    elif no_output:
        return
    else:
        logger.info(f"{dirpath} not empty, and will not be removed.")


def stl2mesh(
    stlfiles,
    meshfile_out,
    resolution,
    subdomain_map=None,
    remove_subdomains=None,
):
    """Creates an svmtk-domain from a set of stl-files, and stores as a meshio .mesh-file. May optionally remove
    sepecifc subdomains, or add subdomain markers."""
    assert isinstance(
        stlfiles, list
    ), "stlfiles should be list. (single surface may be wrapped as length 1 list)."
    surfaces = [svmtk.Surface(str(stl)) for stl in stlfiles]
    if subdomain_map is not None:
        domain = svmtk.Domain(surfaces, subdomain_map)
    else:
        domain = svmtk.Domain(surfaces)
    domain.create_mesh(resolution)
    if remove_subdomains is not None:
        domain.remove_subdomain(remove_subdomains)
    domain.save(str(meshfile_out))


def geo2mesh(infile, outfile, dim):
    os.system(f"gmsh -{dim} -format mesh -o '{outfile}' '{infile}'")


def meshfunction_default_value(meshfunction, value: int = 0):
    """Sets the default value for a MeshFunctionSize_t created from a
    MeshValueCollection"""
    for idx, val in enumerate(meshfunction.array() + 1):
        if val == 0:
            meshfunction[idx] = value
    return meshfunction


def mesh2xdmf(meshfile, xdmfdir, dim):
    mesh = meshio.read(meshfile)
    return meshio2xdmf(mesh, xdmfdir, dim=dim)


def meshio2xdmf(mesh, xdmfdir, dim):
    if dim == 2:
        polytope_label = "triangle"
        facet_label = "line"
    elif dim == 3:
        polytope_label = "tetra"
        facet_label = "triangle"
    else:
        raise ValueError("dim should be in (2, 3), got {}.".format(dim))

    if dim == 2:
        points = prune_z_0(mesh)
    else:
        points = mesh.points
    polytopes = {polytope_label: mesh.cells_dict[polytope_label]}
    facets = {facet_label: mesh.cells_dict[facet_label]}

    # Write the mesh into new xdmf file
    meshdata = meshio.Mesh(points, polytopes)
    meshio.write("{}/mesh.xdmf".format(xdmfdir), meshdata)
    if "gmsh:physical" or "medit:ref" in mesh.cell_data_dict:
        cell_data_name = (
            "gmsh:physical" if "gmsh:physical" in mesh.cell_data_dict else "medit:ref"
        )
        # Write the subdomains of the mesh
        subdomains = {
            "subdomains": [mesh.cell_data_dict[cell_data_name][polytope_label]]
        }
        subdomainfile = meshio.Mesh(points, polytopes, cell_data=subdomains)
        meshio.write("{}/subdomains.xdmf".format(xdmfdir), subdomainfile)

        # Write the boundaries/interfaces of the mesh
        boundaries = {"boundaries": [mesh.cell_data_dict[cell_data_name][facet_label]]}
        boundaryfile = meshio.Mesh(points, facets, cell_data=boundaries)
        meshio.write("{}/boundaries.xdmf".format(xdmfdir), boundaryfile)

    return xdmfdir


def xdmf2hdf(xdmfdir, hdf5file):
    # Read xdmf-file into a FEniCS mesh
    dirpath = Path(xdmfdir)
    mesh = Mesh()
    with XDMFFile(str(dirpath / "mesh.xdmf")) as meshfile:
        meshfile.read(mesh)

    # Read cell data to a MeshFunction (of dim n)
    n = mesh.topology().dim()
    subdomains = MeshFunction("size_t", mesh, n)
    if (dirpath / "subdomains.xdmf").exists():
        with XDMFFile(str(dirpath / "subdomains.xdmf")) as subdomainfile:
            subdomainfile.read(subdomains, "subdomains")

    bdrycollection = MeshValueCollection("size_t", mesh, n - 1)
    if (dirpath / "boundaries.xdmf").exists():
        with XDMFFile(str(dirpath / "boundaries.xdmf")) as boundaryfile:
            boundaryfile.read(bdrycollection, "boundaries")
    boundaries = MeshFunction("size_t", mesh, bdrycollection)

    with HDF5File(mesh.mpi_comm(), str(hdf5file), "w") as f:
        f.write(mesh, "/domain/mesh")
        f.write(subdomains, "/domain/subdomains")
        f.write(boundaries, "/domain/boundaries")
    return hdf5file


def geo2hdf(infile, outfile, dim, tmpdir=None):
    """Single file for creating h5-file from gmsh .geo file."""
    inpath = Path(infile)
    tmpdir = create_tmpdir(tmpdir)
    meshfile = Path(f"{tmpdir}/{inpath.stem}.mesh")
    geo2mesh(inpath, meshfile, dim)
    mesh2xdmf(meshfile, tmpdir, dim)
    xdmf2hdf(tmpdir, outfile)
    return Path(outfile)


def stl2hdf(
    stlfiles,
    outfile,
    resolution,
    subdomain_map=None,
    remove_subdomains=None,
    tmpdir=None,
):
    tmpdir = create_tmpdir(tmpdir)
    logger.info("Intermediated files stored in: ", tmpdir)

    stl2mesh(
        stlfiles,
        tmpdir / "meshfile.mesh",
        resolution,
        subdomain_map=subdomain_map,
        remove_subdomains=remove_subdomains,
    )
    mesh2hdf(
        tmpdir / "meshfile.mesh", outfile, dim=3, tmpdir=tmpdir
    )  # Dont think stls will be relevant in 2-dim case
    return Path(outfile)


def mesh2hdf(infile, outfile, dim, tmpdir=None):
    """Single function for creating an h5-file from a meshfile of given dimension.
    TODO: Write test function"""
    inpath = Path(infile).resolve()
    tmppath = create_tmpdir(tmpdir)
    logger.info(f"Intermediate files stored in: {tmppath}")
    # Go from mesh->xdmf->h5
    mesh2xdmf(inpath, tmppath, dim)
    xdmf2hdf(tmppath, outfile)
    return outfile


def hdf2fenics(
    hdf5file, pack=False
) -> Domain | tuple[Mesh, MeshFunction, MeshFunction]:
    """Function to read h5-file with annotated mesh, subdomains
    and boundaries into fenics mesh"""
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(hdf5file), "r") as hdf:
        hdf.read(mesh, "/domain/mesh", False)
        n = mesh.topology().dim()
        subdomains = MeshFunction("size_t", mesh, n)
        if hdf.has_dataset("/domain/subdomains"):
            hdf.read(subdomains, "/domain/subdomains")
        boundaries = MeshFunction("size_t", mesh, n - 1)
        if hdf.has_dataset("/domain/boundaries"):
            hdf.read(boundaries, "/domain/boundaries")

    if pack:
        return Domain(mesh, subdomains, boundaries)

    return mesh, subdomains, boundaries
