/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "queue.hh"

namespace holovibes
{
namespace worker
{
class InformationWorker;
}

/*!
 * \brief Class used to store informations that will be fetch by the info
 */
class InformationContainer
{
  public:
    using QueueType = Queue::QueueType;

    /*!
     * \brief Indication type that is store in maps
     */
    enum class IndicationType
    {
        IMG_SOURCE,

        INPUT_FORMAT,
        OUTPUT_FORMAT,

        CUTS_SLICE_CURSOR,
    };

    /*!
     * \brief Fps type that is store in maps
     */
    enum class FpsType
    {
        INPUT_FPS,
        OUTPUT_FPS,
        SAVING_FPS,
    };

    /*!
     * \brief Progress type that is store in maps
     */
    enum class ProgressType
    {
        FILE_READ,
        FRAME_RECORD,
        CHART_RECORD,
    };

    /*!
     * \brief InformationContainer Default constructor
     */
    InformationContainer() = default;

    /*!
     * \brief Setter for display_info_text_function attribute
     *
     * \param function Function to set
     */
    void set_display_info_text_function(
        const std::function<void(const std::string&)>& function);

    /*!
     * \brief Setter for update_progress_function attribute
     *
     * \param function Function to set
     */
    void set_update_progress_function(
        const std::function<void(ProgressType, size_t, size_t)>& function);

    /*!
     * \brief Add an indication in the indication_map_
     *
     * \param indication_type Type of the indication to add
     * \param indication Indication to add in the map
     */
    void add_indication(IndicationType indication_type,
                        const std::string& indication);

    /*!
     * \brief Add a new processed fps in the fps_map_
     *
     * \param fps_type Type of the processed fps to add
     * \param processed_fps Processed fps to add
     */
    void add_processed_fps(FpsType fps_type,
                           std::atomic<unsigned int>& processed_fps);

    /*!
     * \brief Add a new queue size in the queue_size_map_
     *
     * \param queue_type Type of the queue that has the size being added in the
     *                   map
     * \param cur_size Current size of the queue
     * \param max_size Maximum size of the queue
     */
    void add_queue_size(QueueType queue_type,
                        const std::atomic<unsigned int>& cur_size,
                        const std::atomic<unsigned int>& max_size);

    /*!
     * \brief Add a new progress type in the progress_index_map_
     *
     * \param progress_type Type of the progress to add
     * \param cur_progress Current progress
     * \param max_progress Maximum progress
     */
    void add_progress_index(ProgressType progress_type,
                            const std::atomic<unsigned int>& cur_progress,
                            const std::atomic<unsigned int>& max_progress);

    /*!
     * \brief Remove an indication from the indication_map_
     *
     * \param info_type Type of the info to remove
     */
    void remove_indication(IndicationType info_type);

    /*!
     * \brief Remove a processed fps from the fps_map_
     * 
     * \param fps_type Type of the processed fps to add
     */
    void remove_processed_fps(FpsType fps_type);

    /*!
     * \brief Remove a queue size from the queue_size_map_
     *
     * \param queue_type Type of the queue that has the size being added in the
     *                   map
     */
    void remove_queue_size(QueueType queue_type);

    /*!
     * \brief Remove a progress type from the progress_index_map_
     *
     * \param progress_type Type of the progress
     */
    void remove_progress_index(ProgressType progress_type);

    /*!
     * \brief Clear all the information
     *
     * \details Clear the maps
     */
    void clear();

  private:
    // Give access to protected members to the InformationWorker
    friend class worker::InformationWorker;

    //! The mutex used to prevent data races
    mutable std::mutex mutex_;

    //! The map associating an indication type with its name
    static const std::unordered_map<IndicationType, std::string>
        indication_type_to_string_;

    //! The map associating an indication type with its information
    std::map<IndicationType, std::string> indication_map_;

    //! The map associating a fps type with its name
    static const std::unordered_map<FpsType, std::string> fps_type_to_string_;

    //! The map associating a fps type with its corresponding value
    std::map<FpsType, std::atomic<unsigned int>*> fps_map_;

    //! The map associating a queue type with its name
    static const std::unordered_map<QueueType, std::string>
        queue_type_to_string_;

    //! The map associating the queue type with its current size
    //! The first element of the pair is the current size
    //! The second element of the pair is the capacity
    std::map<QueueType,
             std::pair<const std::atomic<unsigned int>*,
                       const std::atomic<unsigned int>*>>
        queue_size_map_;

    std::function<void(const std::string&)> display_info_text_function_ =
        [](const std::string&) {};

    //! The map associating the progress type with its current progress
    //! The first element of the pair is the current progress
    //! The second element of the pair is the max progress
    std::unordered_map<ProgressType,
                       std::pair<const std::atomic<unsigned int>*,
                                 const std::atomic<unsigned int>*>>
        progress_index_map_;

    //! The function used to update the progress displayed
    std::function<void(ProgressType, size_t, size_t)>
        update_progress_function_ = [](ProgressType, size_t, size_t) {};
};
} // namespace holovibes

#include "information_container.hxx"