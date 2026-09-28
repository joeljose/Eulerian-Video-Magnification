"""Deprecated entry point, kept for one release: runs `evm.py --gpu`.

The CPU and GPU pipelines are now one implementation in evm.py.
"""

import sys

import evm

if __name__ == '__main__':
    print("Note: evm_cuda.py is deprecated; use `python evm.py --gpu`.",
          file=sys.stderr)
    evm.main(sys.argv[1:] + ['--gpu'])
