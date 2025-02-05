/*! \file noise_filter.cuh
 *  \brief Déclaration des fonctions de filtrage anti bruit.
 *
 *  Ce module propose une fonction simple qui applique un filtre moyenneur 3×3 sur une image
 *  en format flottant, afin de réduire le bruit.
 */

#pragma once

#include "aliases.hh" // Pour les typedefs (ex. uint)
#include "cuda_runtime.h"

/*!
 * \brief Applique un filtre anti bruit simple (moyenneur 3x3) sur une image.
 *
 * L'image est modifiée in situ.
 *
 * \param[in,out] gpu_image Pointeur sur l'image GPU (format float) à filtrer.
 * \param[in] width     Largeur de l'image.
 * \param[in] height    Hauteur de l'image.
 * \param[in] stream    Stream CUDA à utiliser pour l'exécution.
 */
void apply_noise_filter(float* gpu_image, const uint width, const uint height, const cudaStream_t stream);
