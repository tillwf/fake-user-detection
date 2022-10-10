import logging
import pandas as pd

from fake_user_detection.features.feature import Feature


class CategoryInteraction(Feature):

    @classmethod
    def extract_feature(cls, df):
        """
        While real users interact with just a small subset
        of categories in a time window, bots' behaviour are
        not centred into specific categories and interact
        with a vast majority of them.
        """
        logging.info("Computing the feature CategoryInteraction")

        n_category = len(df["Category"].unique())
        nunique_cat = df.groupby("UserId")["Category"].nunique()
        nunique_cat.name = "n_unique_category"
        nunique_cat_proportion = nunique_cat/n_category
        nunique_cat_proportion.name = "n_unique_category_proportion"

        df = nunique_cat.to_frame().join(nunique_cat_proportion.to_frame())
        df.columns = pd.MultiIndex.from_product([
            ['category_interaction'],
            df.columns
        ])

        return df