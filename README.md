# JPEG-XS CUDA DECODER

This project is based on the [source code](https://jpeg.org/jpegxs/software.html) provided by the JPEG organization for secondary development. The code has been written for each module, but code integration, testing, **and modification have not yet been carried out.** Further updates will be made in the future.

## Version and Release History

**08/10/2023**: release first version, this version uses CUDA to redo the code, but does not undergo module testing and integration, which will be resolved in the next version.

## How to build

Due to time constraints, the makefile has not been written yet. Please use the following command to complete the compilation：

```
nvcc bitpacking.c budget.c buf_mgmt.c data_budget.c dwt.c gcli_budget.c gcli_methods.c ids.c image.c mct.c nlt.c packing.c precinct.c precinct_budget.c precinct_budget_table.c pred.c predbuffer.c quant.c rate_control.c sb_weighting.c sig_flags.c sigbuffer.c version.c xs_config.c xs_config_parser.c xs_dec.c xs_markers.c packing.cu xs_dec_main.c file_io.c cmdline_options.c file_sequence.c image_open.c v210.c rgb16.c yuv16.c planar.c uyvy8.c argb.c mono.c ppm.c pgx.c helpers.c mct.cu nlt.cu dwt.cu -o jxs_decoder -w -rdc=true -gencode=arch=compute_XX,code=compute_XX --default-stream per-thread
```

Remember to modify the **“compute_ XX, Code=compute_ XX“** to  the target machine.

## How to use

### Decoding

Same as the official version:

The decoder takes a .jxs file and generates from that a decoded image (many formats are supported, like PNM, YUV, and PGX). To run it, use ``jxs_decoder [options] <codestream.jxs> <output>``.

The decoder provides the following options:

- ``-D``: To dump the actual JPEG XS configuration as a string. The string is a ``;``-separated line of XS settings. This string can be used by the encoder to encode using the exact same configuration. Applying the option twice (``-DD`` or ``-D -D``) will also show the gain and priority values. Outputs in the same syntax as used by the encoder ``-c`` option.
- ``-f <number>``: Tells the decoder to interpret the input and output as a sequence of files (for frame-based video sequencing). The number specifies the first frame index to process. To be used in combination with ``-n``.
- ``-n <number>``: Specifies the total number number of frames to process.
- ``-v``: Output verbose. Use multiple times to increase the verbosity.

Note: When using the ``-D`` option to show the XS configuration of the codestream, it is optional to specify an output file name. In the case that no output is given, the decoder will not decode the codestream (after showing the XS configuration).

Example:

```
jxs_decoder -v woman.jxs out.ppm
```

## At the **end**

Thanks for my former colleagues and JPEG organization. If there are any legal issues, please contact me to delete this code.
