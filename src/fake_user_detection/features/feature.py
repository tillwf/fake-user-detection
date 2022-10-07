class Feature():

	def __init__(self, level="UserId"):
		self.level = level
		self.data = None

	def extract_feature(self, df):
		return df

	def merge_feature(self, df):
		if self.data:
			return df.merge(data, on=level, how="left")