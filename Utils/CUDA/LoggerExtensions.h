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

inline std::ostream& operator<<(std::ostream& os, const uchar2 e) {
    os << "(" << int(e.x) << ", " << int(e.y) << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const uchar3 e) {
    os << "(" << int(e.x) << ", " << int(e.y) << ", " << int(e.z) << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const uchar4 e) {
    os << "(" << int(e.x) << ", " << int(e.y) << ", " << int(e.z) << ", " << int(e.w) << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const int2 e) {
    os << "(" << e.x << ", " << e.y << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const int3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const dim3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const int4 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ", " << e.w << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const longlong2 e) {
    os << "(" << e.x << ", " << e.y << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const longlong3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const longlong4 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ", " << e.w << ")";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const float2 e) {
    os << "(" << e.x << ", " << e.y << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const float3 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const float4 e) {
    os << "(" << e.x << ", " << e.y << ", " << e.z << ", " << e.w << ")";
    return os;
}

#endif
