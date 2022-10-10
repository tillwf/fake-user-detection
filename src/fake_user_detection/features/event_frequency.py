import logging
import pandas as pd

from fake_user_detection.features.feature import Feature


class EventFrequency(Feature):

    @classmethod
    def extract_feature(cls, df):
        """
        A bot tends to produce more click_ad events in a
        time window than a real user
        """
        logging.info("Computing the feature EventFrequency")

        def n_consecutive_click_as(sub_df):
            return (sub_df.reset_index()
                          ["index"]
                          .diff()
                          .ne(1)
                          .cumsum()
                          .value_counts()
                          .max()
                    )

        n_consecutive_click_ad = (df.query("Event == 'click_ad'")
                                    .groupby("UserId")
                                    .apply(n_consecutive_click_as)
                                 )
        n_consecutive_click_ad.name = "n_consecutive_click_ad"
        n_consecutive_click_ad = n_consecutive_click_ad.to_frame()
        n_consecutive_click_ad.columns = pd.MultiIndex.from_product([
            ['event_frequency'],
            n_consecutive_click_ad.columns
        ])

        return n_consecutive_click_ad