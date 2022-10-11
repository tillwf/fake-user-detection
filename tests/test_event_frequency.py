import pandas as pd
from io import StringIO

from fake_user_detection.features.event_frequency import EventFrequency


def test_event_frequency():
    txt = """
           UserId            Event    Category
0      C08F2A0901         click_ad       Phone
1      53E261B0C4         click_ad    Holidays
2      53E261B0C4         click_ad       Phone
3      C14BD59AF5       send_email  Real_State
4      CE5E2BA5EC       send_email        Jobs
"""

    df = pd.read_csv(StringIO(txt),  delim_whitespace=True)

    expected_output = pd.DataFrame({
        "UserId": ["C08F2A0901", "53E261B0C4"],
        "n_consecutive_click_ad": [1, 2]
    }).set_index("UserId")

    output = EventFrequency.extract_feature(df)["event_frequency"]

    pd.testing.assert_frame_equal(output, expected_output, check_like=True)