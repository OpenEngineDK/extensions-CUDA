// Type conversion functions.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OE_CUDA_CONVERT_
#define _OE_CUDA_CONVERT_

#include <Meta/CUDA.h>
#include <sstream>
#include <string>
#include <Math/Vector.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            /**
             * Type conversion functions.
             *
             * @class Convert Convert.h Utils/Convert.h
             */
            class Convert {
            public:

                static string ToString(const uchar4& t){
                    std::ostringstream out;
                    out << "(";
                    out << (unsigned int)t.x << ", ";
                    out << (unsigned int)t.y << ", ";
                    out << (unsigned int)t.z << ", ";
                    out << (unsigned int)t.w << ")";
                    return out.str();
                }
                
                static string ToString(const int2& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ")";
                    return out.str();
                }

                static string ToString(const int4& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ", ";
                    out << t.z << ", ";
                    out << t.w << ")";
                    return out.str();
                }

                static string ToString(const dim3& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ", ";
                    out << t.z << ")";
                    return out.str();
                }

                static string ToString(const float2& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ")";
                    return out.str();
                }

                static string ToString(const float3& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ", ";
                    out << t.z << ")";
                    return out.str();
                }

                static string ToString(const float4& t){
                    std::ostringstream out;
                    out << "(";
                    out << t.x << ", ";
                    out << t.y << ", ";
                    out << t.z << ", ";
                    out << t.w << ")";
                    return out.str();
                }

                static string ToString(int* vl, unsigned int size){
                    int tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << tmp[0];
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << tmp[i];
                    out << "]";
                    return out.str();
                }

                static string ToString(int2* vl, unsigned int size){
                    int2 tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(int2), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << ToString(tmp[0]);
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << ToString(tmp[i]);
                    out << "]";
                    return out.str();
                }

                static string ToString(int4* vl, unsigned int size){
                    int4 tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(int4), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << ToString(tmp[0]);
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << ToString(tmp[i]);
                    out << "]";
                    return out.str();
                }

                static string ToString(float* vl, unsigned int size){
                    float tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(float), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << tmp[0];
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << tmp[i];
                    out << "]";
                    return out.str();
                }

                static string ToString(float2* vl, unsigned int size){
                    float2 tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(float2), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << ToString(tmp[0]);
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << ToString(tmp[i]);
                    out << "]";
                    return out.str();
                }

                static string ToString(float4* vl, unsigned int size){
                    float4 tmp[size];
                    cudaMemcpy(tmp, vl, size * sizeof(float4), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    std::ostringstream out;
                    out << "[" << ToString(tmp[0]);
                    for (unsigned int i = 1; i < size; ++i)
                        out << ", " << ToString(tmp[i]);
                    out << "]";
                    return out.str();
                }
            };

        }
    }
}

#endif
