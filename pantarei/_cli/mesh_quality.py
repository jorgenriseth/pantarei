#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import dolfin as df

import pantarei as pr


def mesh_info(mesh: df.Mesh) -> dict[str, int | float]:
    num_cells = mesh.num_cells()
    num_vertices = mesh.num_vertices()
    min_diameter = 2 * mesh.rmin()
    max_diameter = 2 * mesh.rmax()
    return {
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
    }


def print_mesh_info(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
) -> None:
    print(f"{'Mesh Quality Information':^35}")
    print("-" * 35)
    print(f"{'Number of cells':<20} : {num_cells}")
    print(f"{'Number of vertices':<20} : {num_vertices}")
    print(f"{'Minimum cell diameter':<20} : {min_diameter:.{decimals}g}")
    print(f"{'Maximum cell diameter':<20} : {max_diameter:.{decimals}g}")


def as_latex_table(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
    indentsize: int = 4,
) -> None:
    indent = " " * indentsize
    str_out = (
        latex_header()
        + 3 * indent
        + latex_row(
            num_cells, num_vertices, min_diameter, max_diameter, decimals
        )
        + latex_footer()
    )
    # print(latex_header())
    # print(
    #     3 * indent
    #     + latex_row(
    #         num_cells, num_vertices, min_diameter, max_diameter, decimals
    #     )
    # )
    # print(latex_footer())
    print(str_out)


def latex_header(col_header: Optional[str] = None):
    col_headers = [col_header] * (col_header is not None) + [
        "# Cells",
        "# Vertices",
        "$h_{min}$",
        "$h_{max}$",
    ]
    formatted_col_headers = [r"\textbf{" + x + "}" for x in col_headers]
    return rf"""
\begin{{table}}[h!]
    \centering 
    \begin{{tabular}}{{lcccc}}
        \hline
        {" & ".join(formatted_col_headers)} + \\
        \hline
"""


def latex_row(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
    header: Optional[str] = None,
) -> str:
    str_formatter = [header] * (header is not None) + [
        str(num_cells),
        str(num_vertices),
        f"{min_diameter:.{decimals}f}",
        f"{max_diameter:.{decimals}f}",
    ]
    return " & ".join(str_formatter) + r" \\"


def latex_footer():
    return r"""
        \hline
    \end{tabular}
    \caption{Mesh Information}
    \label{table: mesh info}
\end{table}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the quality of a FEniCS mesh in an HDF file."
    )
    parser.add_argument(
        "path", type=Path, help="Path to the HDF file containing a FEniCS mesh."
    )
    parser.add_argument(
        "--decimals", type=int, help="Cell diameter decimals", default=3
    )
    args = parser.parse_args()
    hdf_file_path = args.path
    if not hdf_file_path.exists():
        raise FileNotFoundError(
            f"The specified HDF file does not exist: {hdf_file_path}"
        )

    mesh = pr.hdf2fenics(hdf_file_path, pack=True)
    mesh_quality = mesh_info(mesh)
    print_mesh_info(**mesh_quality, decimals=args.decimals)
    print()
    print("-- Latex --")
    as_latex_table(**mesh_quality, decimals=args.decimals)


if __name__ == "__main__":
    main()
