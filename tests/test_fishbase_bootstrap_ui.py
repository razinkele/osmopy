from osmose.fishbase import SpecMatch, TraitEstimate
from ui.components import fishbase_bootstrap as fb


def test_apply_writes_selected_traits_to_config():
    """apply_traits writes only ticked traits, into the right sp{idx} keys."""
    cfg = {"simulation.nspecies": "2", "species.name.sp1": "cod"}
    traits = {
        "species.linf": TraitEstimate(110.0, 108, 53.7, 226.0, "cm"),
        "species.k": TraitEstimate(0.163, 108, 0.048, 0.5, "year^-1"),
    }
    selected = {"species.linf"}
    new_cfg = fb.apply_traits(cfg, species_index=1, traits=traits, selected=selected)
    assert new_cfg["species.linf.sp1"] == "110.0"
    assert "species.k.sp1" not in new_cfg
    assert new_cfg is not cfg


def test_review_rows_pairs_current_and_fetched():
    cfg = {"species.linf.sp0": "120"}
    traits = {"species.linf": TraitEstimate(110.0, 108, 53.7, 226.0, "cm")}
    rows = fb.review_rows(cfg, species_index=0, traits=traits)
    row = next(r for r in rows if r["key"] == "species.linf")
    assert row["current"] == "120" and row["fetched"] == 110.0 and row["n"] == 108
    assert row["label"] and row["label"] != "species.linf"


def test_pick_id_is_shiny_safe():
    pid = fb._pick_id(2, "species.length2weight.condition.factor")
    assert "." not in pid
    assert pid == "fb_pick_2_species_length2weight_condition_factor"


def test_candidate_label():
    m = SpecMatch(spec_code=69, scientific_name="Gadus morhua", common_name="Atlantic cod", db="fb")
    assert fb.candidate_label(m) == "Gadus morhua — Atlantic cod [FB]"
    m2 = SpecMatch(spec_code=1, scientific_name="Genus sp", common_name="", db="slb")
    assert fb.candidate_label(m2) == "Genus sp [SLB]"


def test_setup_ui_includes_bootstrap_control():
    from ui.pages.setup import setup_ui

    html = str(setup_ui())
    assert "Bootstrap from FishBase" in html
    assert "fb_fetch" in html      # the inline panel's Fetch button
    assert "fb_species_select" in html  # the species-select output_ui placeholder
