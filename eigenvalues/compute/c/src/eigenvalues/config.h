/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Global configuration parameter */

#ifndef _CONFIG_H_
#define _CONFIG_H_

// should be power of two
#define  MAX_THREADS_BLOCK                256

#define  MAX_SMALL_MATRIX                 512
#define  MAX_THREADS_BLOCK_SMALL_MATRIX   512

#define  MIN_ABS_INTERVAL                 5.0e-37

#endif // #ifndef _CONFIG_H_
