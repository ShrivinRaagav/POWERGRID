import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from dashboard.utils import (
    load_best_model_info, load_model_ranking_df, load_shap_importance_df,
    load_procurement_recommendations_df, load_pareto_front_df,
    load_optimization_summary_df, read_markdown_file, load_image
)

class TestDashboard(unittest.TestCase):
    """Unit test suite for Module 6 Streamlit Dashboard components."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_md = self.temp_dir / "sample.md"
        self.test_md.write_text("# Test Document", encoding="utf-8")

        self.test_img_path = self.temp_dir / "sample.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(self.test_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_best_model_info(self):
        info = load_best_model_info()
        self.assertIn("best_model", info)

    def test_load_model_ranking_df(self):
        df = load_model_ranking_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_load_shap_importance_df(self):
        df = load_shap_importance_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_procurement_recommendations_df(self):
        df = load_procurement_recommendations_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_pareto_front_df(self):
        df = load_pareto_front_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_optimization_summary_df(self):
        df = load_optimization_summary_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_read_markdown_file(self):
        text = read_markdown_file(self.test_md)
        self.assertEqual(text, "# Test Document")

    def test_load_image(self):
        img = load_image(self.test_img_path)
        self.assertIsNotNone(img)

if __name__ == "__main__":
    unittest.main()
