"""Fixes for public-data defects that make the official neuralbench pipeline crash. Import BEFORE building a config.

Alljoined-1.6M (Track 1): some blocks hold groups of up to 5 DIFFERENT images stamped with the same onset
(marker-flush glitch; 160 of 1.66 M stimulus events, measured 2026-10-06). neuralset's segmenter refuses them
("160 triggers of the same type start at the same time on the same timeline"), so the kit's own
`Image[study=xu2025alljoined]` prepare fails. Their true onsets are unknown, so we drop every member of such a group.
`ALLJOINED_FILTER` replaces the overlay's `study.filter_stimuli` step with the same query plus that drop.
"""

import pandas as pd
from neuralset.events.study import EventsTransform

ALLJOINED_QUERY = "type != 'Image' or label in ['stim_train', 'stim_test']"  # = the kit overlay's filter_stimuli


class DropSameOnsetImages(EventsTransform):
    """Apply an optional pandas query, then drop Image events that share (timeline, start) with another one."""

    query: str | None = None

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        from neuralset.events.transforms.basic import query_with_index
        if self.query:
            events = query_with_index(events, self.query)
        img = events[events["type"] == "Image"]
        dup = img.duplicated(["timeline", "start"], keep=False)
        return events.drop(index=dup[dup].index)


ALLJOINED_FILTER = {"study.filter_stimuli": {"name": "DropSameOnsetImages", "query": ALLJOINED_QUERY}}
