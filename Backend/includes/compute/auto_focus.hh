#ifndef AUTO_FOCUS_HH
#define AUTO_FOCUS_HH

#include <cstddef>
#include <cstdint>
#include "API.hh"
#include "frame_desc.hh"
#include "display_queue.hh"

/*!
 * \brief Classe gérant l'autofocus.
 *
 * Elle fournit une méthode de calcul de la métrique de netteté d'une frame
 * ainsi qu'une méthode d'ajustement du focus (z_distance) en fonction de cette métrique.
 */
class AutoFocus
{
  public:
    /*!
     * \brief
     *
     */
    AutoFocus(holovibes::DisplayQueue* q);
    ~AutoFocus();

    /**
     * \brief Ajuste la mise au point en modifiant z_distance.
     *
     * L'algorithme simple compare la métrique courante avec celle de la frame précédente.
     * Si la métrique s'améliore, il continue dans la même direction, sinon il inverse le sens.
     *
     * @param frameData Pointeur sur les données de l'image.
     * @param frameDesc Descripteur de la frame.
     */
    void adjustFocus();

  private:
    float previousMetric_; // Stocke la métrique de la frame précédente
    int adjustDirection_;  // Direction d'ajustement (+1 ou -1)
    float stepSize_;       // Pas d'incrémentation pour z_distance (en mètres)
    holovibes::DisplayQueue* output_;
};

#endif // AUTO_FOCUS_HH
