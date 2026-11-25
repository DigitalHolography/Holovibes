#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <limits>

#include "API.hh"
#include "test_disable_log.hh"

namespace holovibes
{
namespace
{
TEST(LocalPcaSerialization, OptionalGeometryDefaultsAndRoundTrips)
{
    Rendering rendering;
    json legacy = rendering;
    EXPECT_FALSE(legacy.contains("local_pca_width_x"));
    EXPECT_FALSE(legacy.contains("local_pca_rank_factor"));
    rendering.pca_width_x = 7;
    rendering.pca_rank_factor = 2.f;
    from_json(legacy, rendering);
    EXPECT_EQ(rendering.pca_width_x, 32u);
    EXPECT_EQ(rendering.pca_width_y, 32u);
    EXPECT_EQ(rendering.pca_stride_x, 16u);
    EXPECT_EQ(rendering.pca_stride_y, 16u);
    EXPECT_FLOAT_EQ(rendering.pca_rank_factor, 1.f);

    rendering.time_transformation = TimeTransformation::LOCAL_PCA;
    rendering.pca_width_x = 24;
    rendering.pca_width_y = 12;
    rendering.pca_stride_x = 8;
    rendering.pca_stride_y = 3;
    rendering.pca_rank_factor = .75f;
    json encoded = rendering;
    EXPECT_EQ(encoded["time_transformation"], "LOCAL_PCA");
    EXPECT_EQ(encoded["local_pca_width_x"], 24u);
    EXPECT_FALSE(encoded.contains("pca_rank_factor"));
    const auto decoded = encoded.get<Rendering>();
    EXPECT_EQ(decoded.pca_width_x, 24u);
    EXPECT_EQ(decoded.pca_width_y, 12u);
    EXPECT_EQ(decoded.pca_stride_x, 8u);
    EXPECT_EQ(decoded.pca_stride_y, 3u);
    EXPECT_FLOAT_EQ(decoded.pca_rank_factor, .75f);
    encoded.erase("local_pca_rank_factor");
    encoded["pca_rank_factor"] = 2.f;
    EXPECT_FLOAT_EQ(encoded.get<Rendering>().pca_rank_factor, 2.f);
    EXPECT_EQ(json("Local PCA").get<TimeTransformation>(), TimeTransformation::LOCAL_PCA);
}

class LocalPcaSettingsTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            GTEST_SKIP() << "Application settings require CUDA";
        ASSERT_TRUE(API.compute.get_is_computation_stopped());
        original_.Update();
        original_import_ = API.input.get_import_type();
        API.input.set_import_type(ImportType::None);
        initialized_ = true;
        auto baseline = original_;
        baseline.image_rendering.image_mode = Computation::Hologram;
        baseline.image_rendering.time_transformation = TimeTransformation::NONE;
        baseline.image_rendering.space_transformation = SpaceTransformation::NONE;
        baseline.image_rendering.time_transformation_size = 12;
        baseline.image_rendering.batch_size = 4;
        baseline.image_rendering.time_transformation_stride = 4;
        baseline.image_rendering.pca_width_x = 32;
        baseline.image_rendering.pca_width_y = 32;
        baseline.image_rendering.pca_stride_x = 16;
        baseline.image_rendering.pca_stride_y = 16;
        baseline.image_rendering.pca_rank_factor = 1.f;
        baseline.image_rendering.pca_effective_rank_enabled = false;
        baseline.image_rendering.pca_effective_rank_factor = 1.f;
        baseline.advanced.buffer_size.input = 64;
        baseline.view.image_type = ImgType::Modulus;
        baseline.view.z = ViewPQ{};
        baseline.Load();
    }
    void TearDown() override
    {
        if (initialized_)
        {
            original_.Load();
            API.input.set_import_type(original_import_);
        }
    }
    ComputeSettings original_;
    ImportType original_import_ = ImportType::None;
    bool initialized_ = false;
};

TEST_F(LocalPcaSettingsTest, TemporalWindowIsDisjointAndBatchSizeDividesIt)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    EXPECT_EQ(API.transform.get_time_transformation_size(), 12u);
    EXPECT_EQ(API.transform.get_time_stride(), 12u);
    EXPECT_EQ(API.transform.set_time_stride(1), ApiCode::NO_CHANGE);
    EXPECT_EQ(API.transform.set_time_stride(24), ApiCode::NO_CHANGE);
    EXPECT_EQ(API.transform.set_batch_size(5), ApiCode::NO_CHANGE);
    EXPECT_EQ(API.transform.get_batch_size(), 4u);

    EXPECT_EQ(API.transform.set_time_transformation_size(10), ApiCode::OK);
    EXPECT_EQ(API.transform.get_batch_size(), 2u);
    EXPECT_EQ(API.transform.get_time_stride(), 10u);
}

TEST_F(LocalPcaSettingsTest, RejectsGapsAndZeroGeometryWithoutChangingSettings)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    const auto before = API.settings.compute_settings_to_json();
    EXPECT_EQ(API.transform.set_local_pca_parameters(0, 32, 16, 16), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.set_local_pca_parameters(32, 0, 16, 16), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.set_local_pca_parameters(32, 32, 0, 16), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.set_local_pca_parameters(32, 32, 16, 0), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.set_local_pca_parameters(8, 32, 9, 16), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.transform.set_local_pca_parameters(32, 8, 16, 9), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.settings.compute_settings_to_json(), before);
}

TEST_F(LocalPcaSettingsTest, JsonRoundTripAndPartialPatchesPreserveGeometry)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    ASSERT_EQ(API.transform.set_local_pca_parameters(24, 12, 8, 3), ApiCode::OK);
    json saved = API.settings.compute_settings_to_json();
    ASSERT_EQ(API.transform.set_local_pca_parameters(32, 32, 16, 16), ApiCode::OK);
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_EQ(API.transform.get_pca_width_x(), 24u);
    EXPECT_EQ(API.transform.get_pca_width_y(), 12u);
    EXPECT_EQ(API.transform.get_pca_stride_x(), 8u);
    EXPECT_EQ(API.transform.get_pca_stride_y(), 3u);
    EXPECT_EQ(API.transform.get_time_stride(), 12u);

    json patch = {{"version", "v1"}, {"view", {{"fft_shift", false}}}};
    ASSERT_EQ(API.settings.load_compute_settings(patch), ApiCode::OK);
    EXPECT_EQ(API.transform.get_pca_width_x(), 24u);
    EXPECT_EQ(API.transform.get_pca_stride_y(), 3u);

    patch["image_rendering"] = {{"pca_width_x", 20u}, {"pca_stride_y", 4u}, {"local_pca_stride_y", 5u}};
    ASSERT_EQ(API.settings.load_compute_settings(patch), ApiCode::OK);
    EXPECT_EQ(API.transform.get_pca_width_x(), 20u);
    EXPECT_EQ(API.transform.get_pca_stride_x(), 8u);
    EXPECT_EQ(API.transform.get_pca_stride_y(), 5u);

    saved["image_rendering"].erase("local_pca_width_x");
    saved["image_rendering"].erase("local_pca_width_y");
    saved["image_rendering"].erase("local_pca_stride_x");
    saved["image_rendering"].erase("local_pca_stride_y");
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_EQ(API.transform.get_pca_width_x(), 32u);
    EXPECT_EQ(API.transform.get_pca_width_y(), 32u);
    EXPECT_EQ(API.transform.get_pca_stride_x(), 16u);
    EXPECT_EQ(API.transform.get_pca_stride_y(), 16u);
}

TEST_F(LocalPcaSettingsTest, NormalizationPreservesSharedSpatialAndDisplaySettings)
{
    ASSERT_EQ(API.transform.set_space_transformation(SpaceTransformation::FRESNELTR), ApiCode::OK);
    ASSERT_EQ(API.compute.set_img_type(ImgType::SquaredModulus), ApiCode::OK);
    ASSERT_EQ(API.transform.set_p_index(2), ApiCode::OK);
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    EXPECT_EQ(API.transform.get_space_transformation(), SpaceTransformation::FRESNELTR);
    EXPECT_EQ(API.compute.get_img_type(), ImgType::SquaredModulus);
    EXPECT_EQ(API.transform.get_p_index(), 2u);
}

TEST_F(LocalPcaSettingsTest, RankFactorValidatesAndSurvivesSettingsRoundTripAndPartialPatches)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    EXPECT_EQ(API.transform.set_pca_rank_factor(1.f), ApiCode::NO_CHANGE);
    for (const float invalid : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        EXPECT_EQ(API.transform.set_pca_rank_factor(invalid), ApiCode::INVALID_VALUE);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 1.f);

    ASSERT_EQ(API.transform.set_pca_rank_factor(0.f), ApiCode::OK);
    ASSERT_EQ(API.transform.set_pca_rank_factor(75.25f), ApiCode::OK);
    json saved = API.settings.compute_settings_to_json();
    EXPECT_FLOAT_EQ(saved["image_rendering"]["local_pca_rank_factor"].get<float>(), 75.25f);
    ASSERT_EQ(API.transform.set_pca_rank_factor(2.f), ApiCode::OK);
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 75.25f);

    json patch = {{"version", "v1"}, {"view", {{"fft_shift", false}}}};
    ASSERT_EQ(API.settings.load_compute_settings(patch), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 75.25f);

    saved["image_rendering"].erase("local_pca_rank_factor");
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 1.f);
}

TEST_F(LocalPcaSettingsTest, ConfiguredRankFactorAcceptsLargeValues)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    json saved = API.settings.compute_settings_to_json();
    saved["image_rendering"]["local_pca_rank_factor"] = -2.f;
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 0.f);
    saved["image_rendering"]["local_pca_rank_factor"] = 20.f;
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 20.f);

    // Large numeric factors remain valid both through the API and settings loading.
    const float maximum = std::numeric_limits<float>::max();
    EXPECT_EQ(API.transform.set_pca_rank_factor(maximum), ApiCode::OK);
    saved = API.settings.compute_settings_to_json();
    ASSERT_EQ(API.transform.set_pca_rank_factor(1.f), ApiCode::OK);
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), maximum);

    UPDATE_SETTING(PcaRankFactor, std::numeric_limits<float>::quiet_NaN());
    API.transform.normalize_local_pca_settings();
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 1.f);
}

TEST_F(LocalPcaSettingsTest, GlobalEffectiveRankSettingsAreIndependentAndDefaultOffForLegacyPresets)
{
    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::PCA), ApiCode::OK);
    auto legacy = API.settings.compute_settings_to_json();
    EXPECT_FALSE(legacy["image_rendering"].contains("pca_effective_rank_enabled"));

    json patch = {{"version", "v1"},
                  {"image_rendering", {{"pca_effective_rank_enabled", true}, {"pca_effective_rank_factor", 75.25f}}}};
    ASSERT_EQ(API.settings.load_compute_settings(patch), ApiCode::OK);
    EXPECT_TRUE(API.transform.get_pca_effective_rank_enabled());
    EXPECT_FLOAT_EQ(API.transform.get_pca_effective_rank_factor(), 75.25f);

    ASSERT_EQ(API.transform.set_time_transformation(TimeTransformation::LOCAL_PCA), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 1.f);
    ASSERT_EQ(API.transform.set_pca_rank_factor(0.f), ApiCode::OK);
    EXPECT_FLOAT_EQ(API.transform.get_pca_effective_rank_factor(), 75.25f);

    auto saved = API.settings.compute_settings_to_json();
    ASSERT_EQ(API.settings.load_compute_settings(legacy), ApiCode::OK);
    EXPECT_FALSE(API.transform.get_pca_effective_rank_enabled());
    EXPECT_FLOAT_EQ(API.transform.get_pca_effective_rank_factor(), 1.f);
    ASSERT_EQ(API.settings.load_compute_settings(saved), ApiCode::OK);
    EXPECT_TRUE(API.transform.get_pca_effective_rank_enabled());
    EXPECT_FLOAT_EQ(API.transform.get_pca_effective_rank_factor(), 75.25f);
    EXPECT_FLOAT_EQ(API.transform.get_pca_rank_factor(), 0.f);
}
} // namespace
} // namespace holovibes
