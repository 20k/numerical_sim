#ifndef BITFLAGS_CL_INCLUDED
#define BITFLAGS_CL_INCLUDED

enum derivative_bitflags
{
    D_LOW = 1,
    D_FULL = (1 << 1),
    D_ONLY_PX = (1 << 2),
    D_ONLY_PY = (1 << 3),
    D_ONLY_PZ = (1 << 4),
    D_BOTH_PX = (1 << 5),
    D_BOTH_PY = (1 << 6),
    D_BOTH_PZ = (1 << 7),
    D_GTE_WIDTH_1 = (1 << 8),
    D_GTE_WIDTH_2 = (1 << 9),
    D_GTE_WIDTH_3 = (1 << 10),
    D_GTE_WIDTH_4 = (1 << 11),
    D_GTE_WIDTH_5 = (1 << 12),
    D_GTE_WIDTH_6 = (1 << 13)
};

#endif // BITFLAGS_CL_INCLUDED
