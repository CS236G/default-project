import os

from torch.utils.cpp_extension import load


_backend = load(
    name="_metrics_backend",
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["--compiler-bindir=/usr/bin/gcc-8"],
    sources=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", f)
        for f in [
            "nmdistance/nmdistance.cpp",
            "nmdistance/nmdistance.cu",
            "matchcost/matchcost.cpp",
            "matchcost/matchcost.cu",
            "bindings.cpp",
        ]
    ],
)
