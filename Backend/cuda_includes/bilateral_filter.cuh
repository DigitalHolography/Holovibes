#pragma once
#include "cuda_runtime.h"

/*!
 * \brief Applique un filtre bilatéral sur une image.
 *
 * Le filtre bilatéral permet de réduire le bruit tout en préservant les contours,
 * ce qui est particulièrement utile pour la visualisation des structures vasculaires.
 *
 * \param[in,out] gpu_image Pointeur sur l'image GPU (format float) à filtrer.
 * \param[in] width         Largeur de l'image.
 * \param[in] height        Hauteur de l'image.
 * \param[in] sigma_spatial Paramètre sigma pour la composante spatiale.
 * \param[in] sigma_range   Paramètre sigma pour la composante d'intensité.
 * \param[in] stream        Stream CUDA à utiliser.
 */
void apply_bilateral_filter(float* gpu_image,
                            const unsigned int width,
                            const unsigned int height,
                            const float sigma_spatial,
                            const float sigma_range,
                            const cudaStream_t stream);