#include "camera_asi.hh"
#include "camera_logger.hh"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <boost/property_tree/ptree.hpp>

// Pour l'interface ASI, inclure le header de l'SDK
#include <ASICamera2.h>

namespace camera
{

// Constructeur : lecture du fichier ini, chargement des paramètres et initialisation de la caméra
CameraAsi::CameraAsi()
    : Camera("asi.ini")
    , cameraID(0)
    , isInitialized(false)
{
    name_ = "ASI Camera";

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }
    else
    {
        Logger::camera()->error("Impossible d'ouvrir le fichier de configuration asi.ini");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    load_default_params();
    init_camera();
}

// Destructeur : ferme la caméra si elle est ouverte
CameraAsi::~CameraAsi() { shutdown_camera(); }

void CameraAsi::init_camera()
{
    fd_.width = 3840;
    fd_.height = 2160;
    fd_.depth = PixelDepth::Bits8;
    fd_.byteEndian = Endianness::LittleEndian;

    int nbCameras = ASIGetNumOfConnectedCameras();
    if (cameraID >= nbCameras)
    {
        Logger::camera()->error("Camera ID {} invalide. Seules {} caméras détectées.", cameraID, nbCameras);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Récupérer les propriétés de la caméra avant ouverture
    ASI_ERROR_CODE ret = ASIGetCameraProperty(&camInfo, cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Impossible de récupérer les propriétés de la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Ouvrir la caméra
    ret = ASIOpenCamera(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec de l'ouverture de la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Initialiser la caméra
    ret = ASIInitCamera(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec de l'initialisation de la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Pour les caméras trigger, définir le mode en mode normal (video mode)
    if (camInfo.IsTriggerCam == ASI_TRUE)
    {
        ret = ASISetCameraMode(cameraID, ASI_MODE_NORMAL);
        if (ret != ASI_SUCCESS)
        {
            Logger::camera()->error("Échec du passage en mode normal pour la caméra ASI (ID: {})", cameraID);
            throw CameraException(CameraException::NOT_INITIALIZED);
        }
    }

    // Configuration de la ROI : on utilise ici la résolution maximale, binning 1 et format RAW8
    int width = 3840;
    int height = 2160;
    ret = ASISetROIFormat(cameraID, width, height, 1, ASI_IMG_RAW8);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec de la configuration de la ROI pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    // Position de départ à (0,0) (après ASISetROIFormat, qui recentre par défaut)
    ret = ASISetStartPos(cameraID, 0, 0);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec du positionnement de départ pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    isInitialized = true;
    Logger::camera()->info("Caméra ASI (ID: {}) initialisée avec succès", cameraID);
}

// Démarrage de l'acquisition vidéo
void CameraAsi::start_acquisition()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    ASI_ERROR_CODE ret = ASIStartVideoCapture(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec du démarrage de l'acquisition vidéo pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
    Logger::camera()->info("Acquisition vidéo démarrée pour la caméra ASI (ID: {})", cameraID);
}

// Arrêt de l'acquisition vidéo
void CameraAsi::stop_acquisition()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    ASI_ERROR_CODE ret = ASIStopVideoCapture(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec de l'arrêt de l'acquisition vidéo pour la caméra ASI (ID: {})", cameraID);
        // On peut choisir de ne pas lancer d'exception ici, seulement consigner l'erreur.
    }
    Logger::camera()->info("Acquisition vidéo arrêtée pour la caméra ASI (ID: {})", cameraID);
}

// Récupération d'une trame capturée
CapturedFramesDescriptor CameraAsi::get_frames()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Récupération des paramètres ROI actuels
    int width = 0, height = 0, bin = 0;
    ASI_IMG_TYPE imgType;
    ASI_ERROR_CODE ret = ASIGetROIFormat(cameraID, &width, &height, &bin, &imgType);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Échec de la lecture de la ROI pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Calcul de la taille du buffer en fonction du format d'image
    long bufferSize = 0;
    if (imgType == ASI_IMG_RAW8 || imgType == ASI_IMG_Y8)
    {
        bufferSize = width * height;
    }
    else if (imgType == ASI_IMG_RGB24)
    {
        bufferSize = width * height * 3;
    }
    else if (imgType == ASI_IMG_RAW16)
    {
        bufferSize = width * height * 2;
    }
    else
    {
        Logger::camera()->error("Format d'image non supporté pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Allocation d'un buffer pour stocker la trame
    unsigned char* buffer = new unsigned char[bufferSize];
    ret = ASIGetVideoData(cameraID, buffer, bufferSize, 1000); // délai d'attente jusqu'à 1000 ms
    if (ret != ASI_SUCCESS)
    {
        delete[] buffer;
        Logger::camera()->error("Échec de la récupération des données vidéo pour la caméra ASI (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Construction du descripteur de trame (pour cet exemple, une seule région est utilisée)
    CapturedFramesDescriptor desc;
    desc.region1 = buffer;
    desc.count1 = 1; // une seule trame
    desc.region2 = nullptr;
    desc.count2 = 0;

    return desc;
}

// Fermeture de la caméra et libération des ressources
void CameraAsi::shutdown_camera()
{
    if (isInitialized)
    {
        ASI_ERROR_CODE ret = ASICloseCamera(cameraID);
        if (ret != ASI_SUCCESS)
        {
            Logger::camera()->error("Échec de la fermeture de la caméra ASI (ID: {})", cameraID);
        }
        else
        {
            Logger::camera()->info("Caméra ASI (ID: {}) fermée avec succès", cameraID);
        }
        isInitialized = false;
    }
}

// Chargement des paramètres depuis le fichier ini
void CameraAsi::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    // Chargement de l'identifiant de la caméra (si présent), sinon défaut à 0
    cameraID = pt.get<int>("asi.camera_id", 0);
    Logger::camera()->info("Paramètres ini chargés : camera_id = {}", cameraID);
}

// Chargement des paramètres par défaut
void CameraAsi::load_default_params()
{
    // Initialisation par défaut
    std::memset(&camInfo, 0, sizeof(ASI_CAMERA_INFO));
    isInitialized = false;
}

// Liaison (binding) des paramètres internes
void CameraAsi::bind_params()
{
    // Exemple : consigner la résolution maximale de la caméra
    Logger::camera()->info("Résolution maximale de la caméra : {}x{}", camInfo.MaxWidth, camInfo.MaxHeight);
}

// Fonction d'usine pour créer un nouvel objet caméra
ICamera* new_camera_device() { return new CameraAsi(); }

} // namespace camera