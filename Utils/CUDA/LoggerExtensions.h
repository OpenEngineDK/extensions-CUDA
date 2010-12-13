// CUDA Logger extensions
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_LOGGER_EXTENSIONS_H_
#define _CUDA_LOGGER_EXTENSIONS_H_

#include <ostream>

std::ostream& operator<<(std::ostream& os, const int2 e) {
    os << "(" << e.x << ", " << e.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const int3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const int4 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ", " << e.w << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const float2 e) {
    os << "(" << e.x << ", " << e.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const float3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const float4 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ", " << e.w << ")";
    return os;
}

#endif
