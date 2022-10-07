from fake_user_detection.features import Feature


class EventDistribution(Feature):

	def extract_feature(self, df):
		"""
		The distribution of events for a given bot tends
		to be skewed towards click_ad and send_email events,
		because the former can bias our algorithms and the
		latter aims to annoy our sellers.
		"""
		pivot = pd.pivot_table(
			df,
			index="UserId",
			columns="Event",
			aggfunc="count"
		)['Category'].fillna(0)
		n_event = pivot.sum(axis=1)
		pivot_pct = (pivot.div(n_event, axis=0)
			              .add_suffix("_pct"))

		self.data = pivot_pct