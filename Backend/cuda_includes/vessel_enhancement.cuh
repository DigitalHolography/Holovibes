#pragma once
#include <cuda_runtime.h>
#include "aliases.hh" // Pour typedefs, ex. uint

/*!
 * \brief Applique un filtre vesselness (inspiré du filtre de Frangi) sur une image.
 *
 * Pour chaque pixel (sauf en bordure), on calcule une approximation de la matrice Hessienne
 * à l'aide de différences centrales, puis on en déduit les valeurs propres.
 * Si la plus grande valeur propre (en valeur absolue) est négative, on calcule le ratio Ra et l'amplitude S
 * pour obtenir la mesure vesselness selon la formule suivante :
 *
 * \f[
 *   V = \exp\left(-\frac{Ra^2}{2\beta^2}\right) \left(1-\exp\left(-\frac{S^2}{2c^2}\right)\right)
 * \f]
 *
 * avec \f$Ra = \frac{| \lambda_1 |}{| \lambda_2 |}\f$ et \f$S = \sqrt{\lambda_1^2 + \lambda_2^2}\f$.
 * On suppose ici que les vaisseaux apparaissent comme des structures sombres (donc \f$\lambda_2 < 0\f$).
 *
 * \param[in,out] gpu_image Pointeur sur l'image GPU (format float) à traiter.
 * \param[in] width         Largeur de l'image.
 * \param[in] height        Hauteur de l'image.
 * \param[in] beta          Paramètre beta (ex. 0.5f) pour le ratio.
 * \param[in] c             Paramètre c (ex. 15.0f) pour l'amplitude.
 * \param[in] stream        Stream CUDA à utiliser pour l'exécution.
 */
void apply_vesselness_filter(
    float* gpu_image, unsigned int width, unsigned int height, float beta, float c, cudaStream_t stream);
