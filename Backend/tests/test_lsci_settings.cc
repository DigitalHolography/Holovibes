#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <utility>

#include "API.hh"
#include "test_disable_log.hh"

namespace holovibes
{
namespace
{
class LsciSettingsTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "The application settings singleton requires CUDA";

        ASSERT_TRUE(API.compute.get_is_computation_stopped());
        original_.Update();
        original_import_ = API.input.get_import_type();
        original_record_ = API.record.get_record_mode();
        original_cuts_ = API.view.get_cuts_view_enabled();
        original_lens_ = API.view.get_lens_view_enabled();
        original_filter_view_ = API.view.get_filter2d_view_enabled();
        original_unwrap_ = API.transform.get_unwrapping_2d();
        original_z_shift_ = GET_SETTING(ZFFTShift);
        original_convolution_ = GET_SETTING(ConvolutionMatrix);
        initialized_ = true;

        API.input.set_import_type(ImportType::None);
        auto baseline = original_;
        baseline.image_rendering.image_mode = Computation::Hologram;
        baseline.image_rendering.space_transformation = SpaceTransformation::NONE;
        baseline.image_rendering.time_transformation = TimeTransformation::NONE;
        baseline.image_rendering.time_transformation_size = 32;
        baseline.image_rendering.lsci_spatial_window = 5;
        baseline.image_rendering.batch_size = 4;
        baseline.image_rendering.time_transformation_stride = 4;
        baseline.advanced.buffer_size.input = 64;
        baseline.Load();
        API.record.set_record_mode(RecordMode::RAW);
    }

    void TearDown() override
    {
        if (!initialized_)
            return;
        API.input.set_import_type(ImportType::None);
        original_.Load();
        UPDATE_SETTING(RecordMode, original_record_);
        UPDATE_SETTING(CutsViewEnabled, original_cuts_);
        UPDATE_SETTING(LensViewEnabled, original_lens_);
        UPDATE_SETTING(Filter2dViewEnabled, original_filter_view_);
        UPDATE_SETTING(Unwrap2d, original_unwrap_);
        UPDATE_SETTING(ZFFTShift, original_z_shift_);
        UPDATE_SETTING(ConvolutionMatrix, original_convolution_);
        API.input.set_import_type(original_import_);
    }

    bool initialized_ = false;
    ComputeSettings original_;
    ImportType original_import_ = ImportType::None;
    RecordMode original_record_ = RecordMode::NONE;
    bool original_cuts_ = false;
    bool original_lens_ = false;
    bool original_filter_view_ = false;
    bool original_unwrap_ = false;
    bool original_z_shift_ = false;
    std::vector<float> original_convolution_;
};

TEST_F(LsciSettingsTest, SelectionUsesPaperWindowAndKeepsEveryFrame)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    EXPECT_EQ(API.transform.get_time_transformation_size(), 16u);
    EXPECT_EQ(API.transform.get_batch_size(), 4u);
    EXPECT_EQ(API.transform.get_time_stride(), 16u);
    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 5u);
    EXPECT_EQ(API.transform.get_space_transformation(), SpaceTransformation::NONE);
    EXPECT_EQ(API.compute.get_img_type(), ImgType::Modulus);

    EXPECT_EQ(API.transform.set_batch_size(7), ApiCode::NO_CHANGE);
    EXPECT_EQ(API.transform.get_batch_size(), 4u);
    EXPECT_EQ(API.transform.set_time_transformation_size(15), ApiCode::OK);
    EXPECT_EQ(API.transform.get_batch_size(), 3u);
    EXPECT_EQ(API.transform.get_time_stride(), 15u);
    EXPECT_EQ(API.transform.set_time_stride(1), ApiCode::NO_CHANGE);
    EXPECT_EQ(API.transform.get_time_stride(), 15u);
}

TEST_F(LsciSettingsTest, RejectsComplexAndSpectralConsumers)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    for (const auto type : {ImgType::Argument, ImgType::PhaseIncrease, ImgType::Composite, ImgType::SquaredModulus})
        EXPECT_EQ(API.compute.set_img_type(type), ApiCode::WRONG_IMG_TYPE);
    EXPECT_EQ(API.transform.set_space_transformation(SpaceTransformation::FRESNELTR), ApiCode::WRONG_COMP_MODE);
    EXPECT_EQ(API.filter2d.set_filter2d_enabled(true), ApiCode::WRONG_COMP_MODE);
    EXPECT_EQ(API.transform.set_fft_shift_enabled(true), ApiCode::WRONG_COMP_MODE);
    EXPECT_EQ(API.transform.set_unwrapping_2d(true), ApiCode::WRONG_COMP_MODE);
    EXPECT_FALSE(API.view.set_3d_cuts_view(true));
    API.view.set_lens_view(true);
    API.view.set_filter2d_view(true);
    EXPECT_FALSE(API.view.get_lens_view_enabled());
    EXPECT_FALSE(API.view.get_filter2d_view_enabled());
    for (const auto mode : {RecordMode::MOMENTS,
                            RecordMode::CUTS_XZ,
                            RecordMode::CUTS_YZ,
                            RecordMode::OCT_CUBE,
                            RecordMode::OCT_CUBE_FLOAT})
        EXPECT_EQ(API.record.set_record_mode(mode), ApiCode::WRONG_COMP_MODE);
    EXPECT_EQ(API.record.get_record_mode(), RecordMode::RAW);
}

TEST_F(LsciSettingsTest, SelectionUsesDirectIntensityAndSpatialWindowValidatesItsBounds)
{
    ASSERT_EQ(API.transform.set_space_transformation(SpaceTransformation::FRESNELTR), ApiCode::OK);
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    EXPECT_EQ(API.transform.get_space_transformation(), SpaceTransformation::NONE);
    EXPECT_EQ(API.transform.set_space_transformation(SpaceTransformation::ANGULARSP), ApiCode::WRONG_COMP_MODE);

    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 5u);
    EXPECT_EQ(API.transform.set_lsci_spatial_window(9), ApiCode::OK);
    for (const uint invalid : {0u, 2u, 64u, 255u})
        EXPECT_EQ(API.transform.set_lsci_spatial_window(invalid), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 9u);
    EXPECT_EQ(API.transform.set_lsci_spatial_window(1), ApiCode::OK);
    EXPECT_EQ(API.transform.set_lsci_spatial_window(63), ApiCode::OK);
    EXPECT_EQ(API.transform.set_lsci_spatial_window(63), ApiCode::NO_CHANGE);
}

TEST_F(LsciSettingsTest, JsonLoadNormalizesIncompatibleSettingsAfterAllFields)
{
    ComputeSettings requested;
    requested.Update();
    requested.image_rendering.time_transformation = TimeTransformation::LSCI;
    requested.image_rendering.time_transformation_size = 24;
    requested.image_rendering.lsci_spatial_window = 8;
    requested.image_rendering.batch_size = 7;
    requested.image_rendering.time_transformation_stride = 100;
    requested.image_rendering.space_transformation = SpaceTransformation::FRESNELTR;
    requested.image_rendering.filter2d.enabled = true;
    requested.image_rendering.convolution.type = "unused-kernel.txt";
    requested.view.image_type = ImgType::Composite;
    requested.view.fft_shift = true;
    requested.view.renorm = true;
    requested.view.registration.registration_enabled = true;
    requested.view.window.xy.output_image_accumulation = 8;
    requested.view.window.xz.enabled = true;
    requested.view.window.yz.enabled = true;
    requested.view.z.start = 5;
    requested.view.z.width = 4;
    requested.view.z2.start = 3;
    requested.view.z2.width = 2;
    UPDATE_SETTING(CutsViewEnabled, true);
    UPDATE_SETTING(LensViewEnabled, true);
    UPDATE_SETTING(Filter2dViewEnabled, true);
    ASSERT_EQ(API.record.set_record_mode(RecordMode::MOMENTS), ApiCode::OK);

    json encoded = requested;
    EXPECT_EQ(encoded["image_rendering"]["time_transformation"], "LSCI");
    ASSERT_EQ(API.settings.load_compute_settings(encoded), ApiCode::OK);
    EXPECT_EQ(API.transform.get_time_transformation(), TimeTransformation::LSCI);
    EXPECT_EQ(API.transform.get_time_transformation_size(), 24u);
    EXPECT_EQ(API.transform.get_batch_size(), 6u);
    EXPECT_EQ(API.transform.get_time_stride(), 24u);
    EXPECT_EQ(API.compute.get_img_type(), ImgType::Modulus);
    EXPECT_EQ(API.transform.get_space_transformation(), SpaceTransformation::NONE);
    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 7u);
    EXPECT_FALSE(API.filter2d.get_filter2d_enabled());
    EXPECT_FALSE(API.view.get_cuts_view_enabled());
    EXPECT_FALSE(API.view.get_lens_view_enabled());
    EXPECT_FALSE(API.view.get_filter2d_view_enabled());
    EXPECT_FALSE(API.global_pp.get_registration_enabled());
    EXPECT_FALSE(API.global_pp.get_renorm_enabled());
    EXPECT_TRUE(API.global_pp.get_convolution_file_name().empty());
    EXPECT_EQ(API.window_pp.get_accumulation_level(WindowKind::XYview), 1u);
    EXPECT_FALSE(GET_SETTING(XZ).enabled);
    EXPECT_FALSE(GET_SETTING(YZ).enabled);
    EXPECT_EQ(API.transform.get_p_index(), 0u);
    EXPECT_EQ(API.transform.get_p_accu_level(), 0u);
    EXPECT_EQ(API.transform.get_q_index(), 0u);
    EXPECT_EQ(API.transform.get_q_accu_level(), 0u);
    EXPECT_EQ(API.record.get_record_mode(), RecordMode::HOLOGRAM);
}

TEST_F(LsciSettingsTest, OldJsonUsesFivePixelsInsteadOfCurrentCustomWindow)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    ASSERT_EQ(API.transform.set_lsci_spatial_window(9), ApiCode::OK);
    json old = API.settings.compute_settings_to_json();
    old["image_rendering"].erase("lsci_spatial_window");
    ASSERT_EQ(API.settings.load_compute_settings(old), ApiCode::OK);
    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 5u);
}

TEST_F(LsciSettingsTest, ViewOnlyJsonPatchPreservesCustomSpatialWindow)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    ASSERT_EQ(API.transform.set_lsci_spatial_window(7), ApiCode::OK);
    json patch = {{"version", "v1"}, {"view", {{"fft_shift", false}}}};
    ASSERT_EQ(API.settings.load_compute_settings(patch), ApiCode::OK);
    EXPECT_EQ(API.transform.get_lsci_spatial_window(), 7u);
    EXPECT_EQ(API.transform.get_time_transformation_size(), 16u);
}

TEST_F(LsciSettingsTest, JsonNormalizesSpatialWindowBoundsAndRoundTripsCustomValues)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LSCI), ApiCode::OK);
    for (const auto [requested, expected] :
         {std::pair{0u, 1u}, std::pair{2u, 1u}, std::pair{64u, 63u}, std::pair{101u, 63u}, std::pair{11u, 11u}})
    {
        json encoded = API.settings.compute_settings_to_json();
        encoded["image_rendering"]["lsci_spatial_window"] = requested;
        ASSERT_EQ(API.settings.load_compute_settings(encoded), ApiCode::OK);
        EXPECT_EQ(API.transform.get_lsci_spatial_window(), expected);
        EXPECT_EQ(API.settings.compute_settings_to_json()["image_rendering"]["lsci_spatial_window"], expected);
    }
}

TEST(LsciSerialization, DirectDeserializationDefaultsOnlyTheMissingSpatialWindow)
{
    Rendering rendering;
    json legacy = rendering;
    EXPECT_FALSE(legacy.contains("lsci_spatial_window"));
    rendering.lsci_spatial_window = 11;
    from_json(legacy, rendering);
    EXPECT_EQ(rendering.lsci_spatial_window, 5u);

    rendering.lsci_spatial_window = 7;
    json custom = rendering;
    EXPECT_EQ(custom["lsci_spatial_window"], 7u);
    EXPECT_EQ(custom.get<Rendering>().lsci_spatial_window, 7u);

    legacy.erase("batch_size");
    EXPECT_THROW(legacy.get<Rendering>(), json::out_of_range);
}
} // namespace
} // namespace holovibes
