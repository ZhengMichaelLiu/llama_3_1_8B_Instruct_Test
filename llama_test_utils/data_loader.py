"""
Data Loading Module for Benchmarks

Author: Zheng Liu
Date: 2025/06/19
"""

import random
from datasets import load_dataset

LONGBENCH_DATASETS = [
    "samsum", "narrativeqa", "qasper", "triviaqa", "hotpotqa",
    "multifieldqa_en", "multifieldqa_zh", "2wikimqa", "musique",
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "lsht", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "lcc", "repobench-p",
]

class DataLoader:
    """Handles loading of benchmark data (LongBench or custom)."""

    def load_data(self, benchmark: str = 'longbench') -> str:
        """
        Load data from specified benchmark.

        Args:
            benchmark: Name of the benchmark ('longbench' or others).

        Returns:
            A string containing the loaded text.
        """
        if benchmark == 'longbench':
            return self._load_longbench()
        else:
            return self._load_custom()

    def _load_longbench(self) -> str:
        """Load a random text sample from HuggingFace LongBench dataset."""
        dataset_name = random.choice(LONGBENCH_DATASETS)
        print(f"Loading LongBench dataset: {dataset_name}")

        try:
            data = load_dataset("THUDM/LongBench", dataset_name, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return self._load_custom()

        entry = random.choice(data)
        text = entry.get("context") or entry.get("input", "")

        if text:
            print(f"Loaded text length: {len(text)} characters")
            return text

        print("No valid text found in dataset.")
        return self._load_custom()

    def _load_custom(self) -> str:
        """Load custom text."""
        print("Using custom text")
        return """Walt Whitman has somewhere a fine and just distinction between "loving by allowance" and "loving with personal love."
This distinction applies to books as well as to men and women; and in the case of the not very numerous authors who are the objects of the personal affection, it brings a curious consequence with it.
There is much more difference as to their best work than in the case of those others who are loved "by allowance" by convention, and because it is felt to be the right and proper thing to love them.
To some the delightful freshness and humour of Northanger Abbey, its completeness, finish, and entrain, obscure the undoubted critical facts that its scale is small, and its scheme, after all, that of burlesque or parody, a kind in which the first rank is reached with difficulty.
Persuasion, relatively faint in tone, and not enthralling in interest, has devotees who exalt above all the others its exquisite delicacy and keeping.
Sense and Sensibility has perhaps the fewest out-and-out admirers; but it does not want them.
I suppose, however, that the majority of at least competent votes would, all things considered, be divided between Emma and the present book; and perhaps the vulgar verdict (if indeed a fondness for Miss Austen be not of itself a patent of exemption from any possible charge of vulgarity) would go for Emma.
It is the larger, the more varied, the more popular; the author had by the time of its composition seen rather more of the world, and had improved her general, though not her most peculiar and characteristic dialogue; such figures as Miss Bates, as the Eltons, cannot but unite the suffrages of everybody.
On the other hand, I, for my part, declare for Pride and Prejudice unhesitatingly.
It seems to me the most perfect, the most characteristic, the most eminently quintessential of its author's works; and for this contention in such narrow space as is permitted to me, I propose here to show cause.
In the first place, the book (it may be barely necessary to remind the reader) was in its first shape written very early, somewhere about 1796, when Miss Austen was barely twenty-one; though it was revised and finished at Chawton some fifteen years later, and was not published till 1813, only four years before her death.
I do not know whether, in this combination of the fresh and vigorous projection of youth, and the critical revision of middle life, there may be traced the distinct superiority in point of construction, which, as it seems to me, it possesses over all the others.
The plot, though not elaborate, is almost regular enough for Fielding; hardly a character, hardly an incident could be retrenched without loss to the story.
The elopement of Lydia and Wickham is not, like that of Crawford and Mrs. Rushworth, a coup de théâtre; it connects itself in the strictest way with the course of the story earlier, and brings about the denouement with complete propriety.
All the minor passages—the loves of Jane and Bingley, the advent of Mr. Collins, the visit to Hunsford, the Derbyshire tour—fit in after the same unostentatious, but masterly fashion.
There is no attempt at the hide-and-seek, in-and-out business, which in the transactions between Frank Churchill and Jane Fairfax contributes no doubt a good deal to the intrigue of Emma, but contributes it in a fashion which I do not think the best feature of that otherwise admirable book.
I do not know whether the all-grasping hand of the playwright has ever been laid upon Pride and Prejudice; and I dare say that, if it were, the situations would prove not startling or garish enough for the footlights, the character-scheme too subtle and delicate for pit and gallery.
But if the attempt were made, it would certainly not be hampered by any of those loosenesses of construction, which, sometimes disguised by the conveniences of which the novelist can avail himself, appear at once on the stage.
I think, however, though the thought will doubtless seem heretical to more than one school of critics, that construction is not the highest merit, the choicest gift, of the novelist.
It sets off his other gifts and graces most advantageously to the critical eye; and the want of it will sometimes mar those graces—appreciably, though not quite consciously—to eyes by no means ultra-critical.
But a very badly-built novel which excelled in pathetic or humorous character, or which displayed consummate command of dialogue—perhaps the rarest of all faculties—would be an infinitely better thing than a faultless plot acted and told by puppets with pebbles in their mouths.
The characteristics of Miss Austen's humour are so subtle and delicate that they are, perhaps, at all times easier to apprehend than to express, and at any particular time likely to be differently apprehended by different persons.
To me this humour seems to possess a greater affinity, on the whole, to that of Addison than to any other of the numerous species of this great British genus.
The differences of scheme, of time, of subject, of literary convention, are, of course, obvious enough; the difference of sex does not, perhaps, count for much, for there was a distinctly feminine element in "Mr. Spectator," and in Jane Austen's genius there was, though nothing mannish, much that was masculine."""
