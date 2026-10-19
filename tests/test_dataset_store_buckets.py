import numpy as np
import pandas as pd
import pytest

from survey2zarray.stores.dataset_store.buckets import (
    BUCKET_ARRAY,
    BUCKETING_RULE,
    MAX_BUCKETS,
    MIN_BUCKETS,
    SOURCES_PER_STRATUM,
    BucketingRule,
    balance_warnings,
    bucket_order,
    bucketing_columns,
    build_rule,
    choose_bucket_count,
    plan_splits,
    quantile_boundaries,
    read_split_config,
    validate_columns,
)
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.dataset_store.edit import append_sources
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from tests.conftest import IMAGE_DEC, IMAGE_RA, build_test_dataset_store

CUTOUT_ARCSEC = 4.0
SPACING_ARCSEC = 10.0


def catalogue(n_sources=48, columns=8):
    # A grid wide enough that no source falls in another's beam, carrying one
    # discrete column and one continuous one
    rows = []
    for index in range(n_sources):
        row, column = divmod(index, columns)
        rows.append(
            {
                "id": f"src_{index:03d}",
                "ra": IMAGE_RA + (column - columns / 2) * SPACING_ARCSEC / 3600.0,
                "dec": IMAGE_DEC
                + (row - n_sources / columns / 2) * SPACING_ARCSEC / 3600.0,
                "kind": "FRI" if index % 2 else "FRII",
                "extent": float(index),
            }
        )
    return pd.DataFrame(rows)


class TestBucketCount:
    def test_it_follows_the_stratification_target(self):
        # 1400 sources across 2 strata want 20 to a bucket, so 70 of them
        assert choose_bucket_count(1400, 2) == 70

    def test_it_is_capped_where_granularity_stops_helping(self):
        # 100,000 sources across 10 strata could take 1000 buckets; the cap
        # holds it at a fraction accurate to 0.2%
        assert choose_bucket_count(100_000, 10) == MAX_BUCKETS

    def test_a_small_catalogue_gives_up_bucket_size_instead(self):
        # Below the target, a bucket count of one would put everything in one
        # split, so the fractions win
        assert choose_bucket_count(30, 5) == MIN_BUCKETS
        assert choose_bucket_count(4, 1) == 4

    def test_it_is_never_zero(self):
        assert choose_bucket_count(0, 1) == 1

    def test_a_bucket_holds_the_target_where_the_catalogue_affords_it(self):
        n_sources, n_strata = 4000, 3
        count = choose_bucket_count(n_sources, n_strata)
        assert n_sources // count >= SOURCES_PER_STRATUM * n_strata


class TestRule:
    def test_discrete_columns_come_first(self):
        rule = build_rule(catalogue(), n_buckets=8)
        assert rule.columns == ("kind", "extent")
        assert rule.kinds == ("discrete", "continuous")
        assert rule.dtypes == ("str", "float64")

    def test_only_continuous_columns_carry_boundaries(self):
        rule = build_rule(catalogue(), n_buckets=8)
        assert set(rule.boundaries) == {"extent"}
        assert rule.boundaries["extent"] == tuple(
            quantile_boundaries(catalogue()["extent"].to_numpy())
        )

    def test_an_identifier_column_is_left_out(self, caplog):
        frame = catalogue(48)
        frame["name"] = [f"unique_{i}" for i in range(len(frame))]
        with caplog.at_level("WARNING"):
            chosen = bucketing_columns(frame)
        assert "name" not in chosen
        assert any("appears source-specific" in r.message for r in caplog.records)

    def test_position_is_not_bucketed_on_by_default(self):
        assert bucketing_columns(catalogue()) == ["kind", "extent"]

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(KeyError, match="Cannot bucket on column"):
            build_rule(catalogue(), ["colour"])

    def test_it_round_trips_through_store_metadata(self):
        rule = build_rule(catalogue(), n_buckets=8)
        assert BucketingRule.from_dict(rule.as_dict()) == rule


class TestDealing:
    def test_every_bucket_is_a_miniature_of_the_catalogue(self):
        frame = catalogue(48)
        rule = build_rule(frame, n_buckets=8)
        order, buckets = bucket_order(frame, rule)
        dealt = frame.iloc[order].reset_index(drop=True)
        for bucket in range(8):
            kinds = dealt.loc[buckets == bucket, "kind"]
            assert set(kinds) == {"FRI", "FRII"}

    def test_the_store_order_ascends_by_bucket(self):
        frame = catalogue(48)
        _, buckets = bucket_order(frame, build_rule(frame, n_buckets=8))
        assert list(buckets) == sorted(buckets)

    def test_the_same_seed_deals_the_same_way(self):
        frame = catalogue(48)
        rule = build_rule(frame, n_buckets=8, seed=3)
        first, second = bucket_order(frame, rule), bucket_order(frame, rule)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_a_different_seed_deals_differently(self):
        frame = catalogue(48)
        first = bucket_order(frame, build_rule(frame, n_buckets=8, seed=0))[0]
        second = bucket_order(frame, build_rule(frame, n_buckets=8, seed=1))[0]
        assert not np.array_equal(first, second)

    def test_rows_are_shuffled_within_each_bucket(self):
        # Without it a bucket holds one classification then the other, and a
        # shard of it is a homogeneous clump rather than a sample
        frame = catalogue(96, columns=12)
        rule = build_rule(frame, n_buckets=8)
        order, buckets = bucket_order(frame, rule)
        dealt = frame.iloc[order].reset_index(drop=True)
        sorted_within = 0
        for bucket in range(8):
            extent = dealt.loc[buckets == bucket, "extent"].to_numpy()
            if list(extent) == sorted(extent):
                sorted_within += 1
        assert sorted_within == 0

    def test_the_deal_carries_on_from_where_it_left_off(self):
        frame = catalogue(48)
        rule = build_rule(frame, n_buckets=8)
        _, buckets = bucket_order(frame.iloc[:8], rule, start_rank=48)
        assert sorted(buckets) == list(range(8))


class TestAppendValidation:
    def test_a_missing_column_is_refused(self):
        frame = catalogue(16)
        rule = build_rule(frame, n_buckets=8)
        with pytest.raises(ValueError, match="missing column"):
            validate_columns(frame.drop(columns=["kind"]), rule)

    def test_a_retyped_column_is_refused(self):
        frame = catalogue(16)
        rule = build_rule(frame, n_buckets=8)
        retyped = frame.copy()
        retyped["extent"] = retyped["extent"].astype(int)
        with pytest.raises(ValueError, match="arrives as int64 and discrete"):
            validate_columns(retyped, rule)

    def test_a_column_the_rule_does_not_name_is_ignored(self):
        frame = catalogue(16)
        rule = build_rule(frame, n_buckets=8)
        extra = frame.copy()
        extra["colour"] = "red"
        validate_columns(extra, rule)


class TestSplitConfig:
    def test_it_reads_the_three_fractions_and_a_seed(self):
        fractions, seed = read_split_config(
            {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 7}
        )
        assert fractions == {"train": 0.8, "val": 0.1, "test": 0.1}
        assert seed == 7

    def test_a_name_left_out_takes_zero(self):
        fractions, _ = read_split_config({"train": 0.8, "test": 0.2, "seed": 0})
        assert fractions["val"] == 0.0

    def test_the_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            read_split_config({"train": 0.8, "test": 0.1, "seed": 0})

    def test_the_names_are_fixed(self):
        with pytest.raises(ValueError, match="Unknown key"):
            read_split_config({"train": 1.0, "holdout": 0.0, "seed": 0})

    def test_it_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="Expected a mapping"):
            read_split_config("train")


class TestCutting:
    def plan(self, n_sources=1000, n_buckets=50, eligible=None, **fractions):
        rng = np.random.default_rng(0)
        buckets = rng.integers(0, n_buckets, n_sources)
        if eligible is None:
            eligible = np.ones(n_sources, dtype=bool)
        section = {"train": 0.8, "val": 0.1, "test": 0.1, **fractions}
        return buckets, eligible, plan_splits(buckets, eligible, section, 0, n_buckets)

    def test_the_fractions_land_within_one_bucket(self):
        _, _, plan = self.plan()
        served = sum(plan.counts.values())
        for name, wanted in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            assert abs(plan.counts[name] / served - wanted) <= 1 / 50

    def test_they_are_of_the_filtered_population(self):
        # Half the sources filtered out, and the shares still hold on what is
        # left rather than on what the store was built from
        n_sources = 1000
        eligible = np.zeros(n_sources, dtype=bool)
        eligible[: n_sources // 2] = True
        _, _, plan = self.plan(n_sources=n_sources, eligible=eligible)
        served = sum(plan.counts.values())
        assert served == n_sources // 2
        assert abs(plan.counts["train"] / served - 0.8) <= 1 / 50

    def test_roles_run_contiguously(self):
        _, _, plan = self.plan()
        walked = [plan.roles[bucket] for bucket in plan.order]
        assert walked == sorted(walked, key=["train", "val", "test"].index)

    def test_the_seed_rotates_where_the_cuts_fall(self):
        buckets = np.random.default_rng(0).integers(0, 50, 1000)
        eligible = np.ones(1000, dtype=bool)
        section = {"train": 0.8, "val": 0.1, "test": 0.1}
        first = plan_splits(buckets, eligible, section, 0, 50)
        second = plan_splits(buckets, eligible, section, 17, 50)
        assert first.order[0] != second.order[0]
        assert not np.array_equal(first.roles, second.roles)

    def test_a_zero_share_takes_no_bucket(self):
        _, _, plan = self.plan(train=0.9, val=0.0, test=0.1)
        assert plan.counts["val"] == 0
        assert "val" not in set(plan.roles)

    def test_every_source_falls_in_exactly_one_split(self):
        buckets, eligible, plan = self.plan()
        served = plan.of_source(buckets)[eligible]
        assert sum(plan.counts.values()) == len(served)


class TestBalanceWarnings:
    def frame(self, n_sources=400):
        rng = np.random.default_rng(0)
        return {
            "kind": np.where(rng.random(n_sources) < 0.5, "FRI", "FRII"),
            "extent": rng.normal(10.0, 2.0, n_sources),
        }

    def rule(self):
        return BucketingRule(
            columns=("kind", "extent"),
            dtypes=("str", "float64"),
            kinds=("discrete", "continuous"),
            boundaries={"extent": ()},
            n_buckets=20,
            seed=0,
        )

    def test_a_representative_split_says_nothing(self):
        columns = self.frame()
        served = np.ones(400, dtype=bool)
        split = np.where(np.arange(400) % 5, "train", "test")
        assert balance_warnings(columns, self.rule(), served, split) == []

    def test_a_skewed_split_is_reported(self):
        columns = self.frame()
        served = np.ones(400, dtype=bool)
        # Every FRI into train and the rest into test, which no bucket cut
        # would produce but a filter aimed at one class would
        split = np.where(columns["kind"] == "FRI", "train", "test")
        reports = balance_warnings(columns, self.rule(), served, split)
        assert any("kind" in line for line in reports)

    def test_a_continuous_column_is_tested_too(self):
        columns = self.frame()
        served = np.ones(400, dtype=bool)
        split = np.where(columns["extent"] < 10.0, "train", "test")
        reports = balance_warnings(columns, self.rule(), served, split)
        assert any("extent" in line for line in reports)

    def test_variable_length_strings_are_compared_at_their_own_dtype(self):
        # What a store reads back. Casting these to a fixed width str raises,
        # so the comparison has to leave the dtype alone
        columns = self.frame()
        columns["kind"] = np.array(list(columns["kind"]), dtype=np.dtypes.StringDType())
        served = np.ones(400, dtype=bool)
        split = np.where(columns["kind"] == "FRI", "train", "test")
        reports = balance_warnings(columns, self.rule(), served, split)
        assert any("kind" in line for line in reports)

    def test_a_representative_split_of_them_says_nothing(self):
        columns = self.frame()
        columns["kind"] = np.array(list(columns["kind"]), dtype=np.dtypes.StringDType())
        served = np.ones(400, dtype=bool)
        split = np.where(np.arange(400) % 5, "train", "test")
        assert balance_warnings(columns, self.rule(), served, split) == []


class TestStoredRule:
    @pytest.fixture
    def store_path(self, tmp_path):
        from tests.conftest import make_continuum_fits

        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_fits(fits_dir / "wide.fits", shape_yx=(200, 200), seed=7)
        survey = tmp_path / "survey"
        build_survey_store(fits_dir, survey, mode="copy")
        path = tmp_path / "catalogue.csv"
        catalogue(48).to_csv(path, index=False)
        out = tmp_path / "dataset"
        build_test_dataset_store(
            [survey], path, CUTOUT_ARCSEC, out, bucket_by=None, n_buckets=8
        )
        return out, tmp_path

    def test_the_rule_and_the_assignment_are_both_stored(self, store_path):
        out, _ = store_path
        root = open_store(out, mode="r")
        assert dict(root.attrs[BUCKETING_RULE])["columns"] == ["kind", "extent"]
        assert root["sources"][BUCKET_ARRAY].shape == (48,)

    def test_the_bucket_column_is_not_served_as_a_catalogue_column(self, store_path):
        out, _ = store_path
        store = DatasetStore.open(out)
        assert set(store.extra_columns) == {"kind", "extent"}
        assert store.bucketing.n_buckets == 8

    def test_appending_leaves_existing_buckets_where_they_were(self, store_path):
        out, tmp_path = store_path
        before = DatasetStore.open(out).buckets.copy()
        new = catalogue(60).iloc[48:].copy()
        new["id"] = [f"new_{i:03d}" for i in range(len(new))]
        append_sources(out, new)
        after = DatasetStore.open(out).buckets
        np.testing.assert_array_equal(before, after[: len(before)])

    def test_an_append_is_bucketed_against_the_stored_boundaries(self, store_path):
        # Recomputing them over the appended catalogue alone would put the same
        # source in a different bin, and the seed would stop reproducing
        out, _ = store_path
        stored = DatasetStore.open(out).bucketing
        new = catalogue(60).iloc[48:].copy()
        new["id"] = [f"new_{i:03d}" for i in range(len(new))]
        recomputed = build_rule(
            new, list(stored.columns), seed=stored.seed, n_buckets=stored.n_buckets
        )
        assert recomputed.boundaries["extent"] != stored.boundaries["extent"]
        assert not np.array_equal(
            stored.sort_keys(new)[1], recomputed.sort_keys(new)[1]
        )

    def test_the_rule_is_read_back_rather_than_worked_out_again(self, store_path):
        out, _ = store_path
        root = open_store(out, mode="r")
        assert DatasetStore.open(out).bucketing == BucketingRule.from_dict(
            dict(root.attrs[BUCKETING_RULE])
        )

    def test_appending_the_same_batch_twice_places_it_the_same_way(self, store_path):
        out, tmp_path = store_path
        new = catalogue(60).iloc[48:].copy()
        new["id"] = [f"new_{i:03d}" for i in range(len(new))]
        stored = DatasetStore.open(out).bucketing
        first = bucket_order(new, stored, start_rank=48)[1]
        second = bucket_order(new, stored, start_rank=48)[1]
        np.testing.assert_array_equal(first, second)

    def test_the_store_order_within_a_bucket_is_not_the_sort_key(self, store_path):
        # A shard holds a run of one bucket, so leaving that run in sort order
        # would give it one classification and a narrow band of extent
        out, _ = store_path
        store = DatasetStore.open(out)
        extent = store.extra_columns["extent"]
        sorted_runs = 0
        for bucket in range(store.bucketing.n_buckets):
            run = extent[store.buckets == bucket]
            if list(run) == sorted(run):
                sorted_runs += 1
        assert sorted_runs == 0

    def test_the_stored_columns_can_be_balance_tested(self, store_path):
        # The columns reach `balance_warnings` from the store rather than from
        # a frame, so this is the path that carries the store's own dtypes
        out, _ = store_path
        store = DatasetStore.open(out)
        served = np.ones(store.n_sources, dtype=bool)
        split = np.where(np.arange(store.n_sources) % 4, "train", "test")
        assert (
            balance_warnings(store.extra_columns, store.bucketing, served, split) == []
        )

    def test_an_append_missing_a_bucketing_column_is_refused(self, store_path):
        out, _ = store_path
        new = catalogue(60).iloc[48:].drop(columns=["extent"]).copy()
        new["id"] = [f"new_{i:03d}" for i in range(len(new))]
        with pytest.raises(ValueError, match="missing column"):
            append_sources(out, new)
