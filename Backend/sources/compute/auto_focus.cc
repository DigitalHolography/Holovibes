#include "laplacian_kernels.cuh"
#include "auto_focus.hh"
#include "API.hh"
#include "common_api.hh"
#include "logger.hh"
#include <cuda_runtime.h>

AutoFocus::AutoFocus(holovibes::DisplayQueue* q)
    : previousMetric_(0.0f)
    , adjustDirection_(1)
    , stepSize_(0.1f)
    , output_(q)
{
}

AutoFocus::~AutoFocus() {}

// Cette fonction ajuste le focus en récupérant la dernière image via l'output de l'API.
// On calcule la métrique de netteté (variance laplacienne) en CUDA et on ajuste la
// distance de focus en inversant la direction si la métrique ne s'améliore pas.
// La nouvelle valeur de z_distance est contrainte entre 1 et 1000.
void AutoFocus::adjustFocus()
{
    // Récupération de l'output depuis la file d'entrée de l'API.

    void* frame = output_->get_last_image();
    if (!frame)
    {
        return;
    }
    LOG_ERROR("debug");

    const camera::FrameDescriptor& fd = output_->get_fd();
    // Calcul de la métrique (variance du Laplacien) via CUDA.
    float currentMetric = processFrameCUDA(frame, fd.width, fd.height, fd.depth);

    LOG_ERROR(currentMetric);
    LOG_ERROR(previousMetric_);

    // Si la métrique ne s'améliore pas, inverser la direction d'ajustement.
    if (currentMetric < previousMetric_)
    {
        adjustDirection_ = -adjustDirection_;
    }
    previousMetric_ = currentMetric;

    // Récupérer la distance actuelle de focus.
    float currentZ = API.transform.get_z_distance() * 1000;
    LOG_ERROR(currentZ);

    // Calculer la nouvelle distance en fonction de la direction et du pas.
    float newZ = currentZ + adjustDirection_ * stepSize_ * 10;

    // Contraindre la nouvelle valeur dans l'intervalle [1, 1000].
    if (newZ < 1.0f)
        newZ = 1.0f;
    if (newZ > 1000.0f)
        newZ = 1000.0f;

    // Appliquer la nouvelle distance de focus via l'API.
    API.transform.set_z_distance(newZ / 1000.0f);
    // Vous pouvez ajouter ici une gestion d'erreur en fonction de 'status'
}