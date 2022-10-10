import logging
import pandas as pd

from fake_user_detection.features.feature import Feature


class EventDistribution(Feature):

    @classmethod
    def extract_feature(cls, df):
        """
        The distribution of events for a given bot tends
        to be skewed towards click_ad and send_email events,
        because the former can bias our algorithms and the
        latter aims to annoy our sellers.
        """
        logging.info("Computing the feature EventDistribution")
        pivot = pd.pivot_table(
            df,
            index="UserId",
            columns="Event",
            aggfunc="count"
        )['Category'].fillna(0)
        n_event = pivot.sum(axis=1)
        pivot_pct = (pivot.div(n_event, axis=0)
                          .add_suffix("_pct"))

        return pivot_pct