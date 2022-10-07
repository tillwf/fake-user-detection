from fake_user_detection.features import Feature


class CategoryInteraction(Feature):

	def extract_feature(self, df):
		"""
		While real users interact with just a small subset
		of categories in a time window, bots' behaviour are
		not centred into specific categories and interact
		with a vast majority of them.
		"""
		n_category = len(df["Category"].unique())
		nunique_cat = df.groupby("UserId")["Category"].nunique()
		nunique_cat.name = "n_unique_category"
		nunique_cat_proportion = nunique_cat/n_category
		nunique_cat_proportion.name = "n_unique_category_proportion"

		self.data = nunique_cat.to_frame().join(nunique_cat_proportion.to_frame())