def test_scenarios_page_has_new_scenario_button_and_quick_duplicate():
    import ui.pages.scenarios as sc

    assert hasattr(sc, "scenarios_ui") and hasattr(sc, "scenarios_server")
    html = str(sc.scenarios_ui())
    assert "btn_new_scenario" in html
    assert "New Scenario" in html
    assert "Quick Duplicate" in html  # Fork relabelled
    assert ">Fork<" not in html  # old label gone
