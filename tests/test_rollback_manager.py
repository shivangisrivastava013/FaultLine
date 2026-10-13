"""
Unit tests for Model Rollback Manager & Smoke Checks.
"""

import os
import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.incidents.database import Base
from src.models.rollback_manager import ModelRollbackManager
from src.models.pipeline import save_model_artifact, create_tree_pipeline, ALL_FEATURES


class BrokenPipelineMock:
    """A broken pipeline that raises an exception during inference."""
    def predict(self, df):
        raise RuntimeError("Fatal tensor crash inside candidate pipeline")


@pytest.fixture
def artifacts_setup(tmp_path):
    artifacts_dir = tmp_path / "models"
    # Create healthy champion model
    champ_dir = artifacts_dir / "champion_v1"
    os.makedirs(champ_dir, exist_ok=True)
    pipe = create_tree_pipeline(max_iter=5)
    # Fit on small dummy data
    df_fit = pd.DataFrame([{
        "distance_km": 5.0,
        "traffic_level": "medium",
        "weather": "clear",
        "hour_of_day": 12,
        "restaurant_load": 0.5,
        "delivery_duration_minutes": 25.0,
    }] * 10)
    pipe.fit(df_fit[ALL_FEATURES], df_fit["delivery_duration_minutes"])
    save_model_artifact(pipe, {"model_version": "v1", "model_type": "tree"}, str(champ_dir))

    # Also save as active
    act_dir = artifacts_dir / "active"
    save_model_artifact(pipe, {"model_version": "v1", "model_type": "tree"}, str(act_dir))

    # Create broken candidate model
    broken_dir = artifacts_dir / "broken_v3"
    save_model_artifact(BrokenPipelineMock(), {"model_version": "v3_broken", "model_type": "broken"}, str(broken_dir))

    return str(artifacts_dir)


def test_failed_smoke_check_preserves_active_model(artifacts_setup):
    manager = ModelRollbackManager(artifacts_dir=artifacts_setup)
    manager.initialize_active_model(default_version="champion_v1")

    assert manager.active_version == "v1"

    # Attempt to activate broken candidate model
    with pytest.raises(RuntimeError) as exc_info:
        manager.activate_version(
            target_version="broken_v3",
            actor="test_user",
            reason="Testing broken model reject",
        )

    assert "smoke check failed" in str(exc_info.value).lower()
    # CRITICAL: Active model MUST still be healthy v1!
    assert manager.active_version == "v1"


def test_successful_rollback_restores_champion(artifacts_setup):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    manager = ModelRollbackManager(artifacts_dir=artifacts_setup)
    manager.initialize_active_model(default_version="champion_v1")

    res = manager.activate_version(
        target_version="champion_v1",
        actor="lead_sre",
        reason="Manual promotion",
        db=db,
    )
    assert res["success"] is True
    assert res["active_version"] == "v1"
    assert res["smoke_test_passed"] is True
    db.close()
