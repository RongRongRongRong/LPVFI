from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_kernel",
    ext_modules=[
        CUDAExtension(
            "my_kernel",
            ["kernel.cpp", "kernel_cu.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)