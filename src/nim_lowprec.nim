## nim_lowprec — low-precision floating-point primitives for ML inference.
##
## Umbrella module: `import nim_lowprec` pulls in the whole family. To take just
## one dtype, import its module directly, e.g. `import nim_lowprec/bfloat16`.

import nim_lowprec/common
import nim_lowprec/bfloat16

export common, bfloat16
