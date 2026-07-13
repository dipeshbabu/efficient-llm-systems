#!/usr/bin/env python3
"""TurboQuant Needle-In-A-Haystack (NIAH) Benchmark v2

Industry-standard NIAH test following Kamradt (2023) and NVIDIA RULER (2024)
methodology for evaluating KV cache compression quality.

Methodology:
  - Kamradt: github.com/gkamradt/LLMTest_NeedleInAHaystack
  - RULER: github.com/NVIDIA/RULER (COLM 2024, arXiv:2404.06654)
  - OpenCompass NeedleBench: opencompass.readthedocs.io

Modes:
  single     Kamradt single-needle: sweep depth (0-100%) x context length
  multi-key  RULER MK-NIAH: real needle + distractors, sweep context length
  multi-value RULER MV-NIAH: multiple same-key needles, sweep length x count

Usage:
    python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf
    python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode single
    python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode multi-key
    python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode multi-value
    python3 tools/validation/niah_test.py --help

Requirements: Python 3.10+ stdlib only (no pip deps).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

# 24 diverse filler paragraphs (~150-200 words each) covering different topics.
# These simulate real essay text a la Paul Graham essays (the de facto NIAH
# standard), but are original to avoid copyright issues.  Each paragraph is on
# a distinct topic so the haystack never feels repetitive within a single test.
FILLER_PARAGRAPHS = [
    # 1 — Astronomy
    (
        "The observable universe spans roughly 93 billion light-years in diameter, "
        "a figure that continues to grow as space itself expands. Within this volume "
        "lie an estimated two trillion galaxies, each hosting hundreds of billions of "
        "stars. Our own Milky Way is a barred spiral galaxy approximately 100,000 "
        "light-years across, containing between 100 and 400 billion stars. The Sun, "
        "an unremarkable yellow dwarf, orbits the galactic center at about 230 "
        "kilometers per second, completing one full revolution every 225 to 250 "
        "million years. Despite the staggering numbers, the universe is overwhelmingly "
        "empty: if you shrank the Sun to the size of a grain of sand, the nearest "
        "star would be roughly four miles away. This vast emptiness is punctuated "
        "by gravitational wells that shape the large-scale structure of the cosmos "
        "into filaments, walls, and voids that stretch across hundreds of millions "
        "of light-years."
    ),
    # 2 — Roman engineering
    (
        "Roman engineers perfected the art of concrete construction over two thousand "
        "years ago, and many of their structures still stand today. The Pantheon in "
        "Rome, completed around 125 AD, features an unreinforced concrete dome that "
        "remains the world's largest of its kind, spanning 43.3 meters. The secret "
        "lay in their mixture: volcanic ash from Pozzuoli combined with lime and "
        "seawater created a remarkably durable material now called pozzolanic "
        "concrete. Modern researchers have discovered that seawater actually "
        "strengthened the material over time by promoting the growth of interlocking "
        "mineral crystals within the matrix. Roman aqueducts, another engineering "
        "marvel, carried water across vast distances using gravity alone, maintaining "
        "a gentle downward slope of roughly one meter per kilometer. The Pont du "
        "Gard in southern France stands 49 meters high and carried water 50 "
        "kilometers from its source to the city of Nimes."
    ),
    # 3 — Deep-sea biology
    (
        "The hadal zone, comprising ocean trenches deeper than 6,000 meters, "
        "represents one of the least explored environments on Earth. Despite "
        "crushing pressures exceeding 1,000 atmospheres, thriving ecosystems exist "
        "in these depths. Snailfish have been observed at nearly 8,200 meters in "
        "the Mariana Trench, making them the deepest-living fish known to science. "
        "Amphipods, small crustaceans related to shrimp, have been collected from "
        "the very bottom of the Challenger Deep at 10,994 meters. These organisms "
        "have evolved specialized proteins and cell membranes that remain functional "
        "under extreme pressure. Hydrothermal vents, first discovered in 1977 near "
        "the Galapagos Rift, support dense communities of tube worms, clams, and "
        "shrimp that derive energy from chemosynthesis rather than photosynthesis. "
        "The discovery fundamentally changed our understanding of where life can "
        "exist and fueled speculation about life on ocean worlds like Europa."
    ),
    # 4 — Printing and literacy
    (
        "The invention of movable type by Johannes Gutenberg around 1440 triggered "
        "a revolution in the dissemination of knowledge across Europe. Before the "
        "printing press, a single book could take months to copy by hand, and "
        "libraries of a few hundred volumes were considered extraordinary. Within "
        "fifty years of Gutenberg's innovation, an estimated twenty million volumes "
        "had been printed, covering subjects from theology and law to astronomy and "
        "medicine. Literacy rates climbed steadily as printed material became "
        "affordable for the emerging middle class. Martin Luther's Ninety-Five "
        "Theses, posted in 1517, spread across Germany in weeks thanks to the press, "
        "accelerating the Protestant Reformation. By the seventeenth century, "
        "newspapers and periodicals had appeared in major European cities, creating "
        "a public sphere of discourse that scholars consider a precondition for "
        "modern democracy. The parallels with today's digital information revolution "
        "are striking, if imperfect."
    ),
    # 5 — Plate tectonics
    (
        "The theory of plate tectonics, formalized in the 1960s, explains the "
        "large-scale motions of Earth's lithosphere. The outer shell of the planet "
        "is divided into roughly fifteen major plates that float on the semi-fluid "
        "asthenosphere below. These plates move at rates comparable to the growth "
        "of a human fingernail, typically two to ten centimeters per year. Where "
        "plates diverge, magma wells up to create new ocean floor, as seen along "
        "the Mid-Atlantic Ridge. Where they converge, one plate may subduct beneath "
        "another, generating deep ocean trenches and volcanic arcs like the Andes. "
        "Transform boundaries, where plates slide past each other horizontally, "
        "produce devastating earthquakes; the San Andreas Fault in California is "
        "the most famous example. Plate tectonics also drives the supercontinent "
        "cycle: roughly every 400 to 600 million years, the continents merge into "
        "a single landmass before breaking apart again."
    ),
    # 6 — Coffee cultivation
    (
        "Coffee cultivation began in the highlands of Ethiopia, where the Coffea "
        "arabica plant still grows wild in montane forests. According to legend, "
        "a goat herder named Kaldi noticed his animals became unusually energetic "
        "after eating red berries from a certain shrub. By the fifteenth century, "
        "coffee was being roasted and brewed in Yemen, and coffeehouses had become "
        "centers of intellectual exchange across the Ottoman Empire. The Dutch "
        "smuggled coffee plants to Java in the late 1600s, breaking the Arab "
        "monopoly and establishing plantations across Southeast Asia. Today, "
        "Brazil produces roughly a third of the world's coffee, followed by Vietnam, "
        "Colombia, and Indonesia. A single coffee tree yields about two thousand "
        "cherries per year, producing roughly one pound of roasted beans. Climate "
        "change threatens arabica production, as the plant requires specific "
        "temperature and rainfall conditions found in a narrow tropical band."
    ),
    # 7 — Music theory
    (
        "Western music theory traces its roots to Pythagoras, who discovered that "
        "harmonious intervals correspond to simple numerical ratios of string "
        "lengths. An octave is a 2:1 ratio, a perfect fifth 3:2, and a perfect "
        "fourth 4:3. This insight laid the groundwork for the mathematical study "
        "of harmony that persists to this day. The twelve-tone equal temperament "
        "system, adopted widely in the eighteenth century, divides the octave into "
        "twelve equal semitones, each separated by a frequency ratio of the twelfth "
        "root of two. This compromise sacrifices the purity of Pythagorean intervals "
        "but allows instruments to play in any key without retuning. Johann "
        "Sebastian Bach's Well-Tempered Clavier, composed in 1722, demonstrated the "
        "versatility of this system with preludes and fugues in all twenty-four "
        "major and minor keys. Modern electronic music has pushed beyond these "
        "boundaries, exploring microtonal scales and algorithmic composition."
    ),
    # 8 — Glacier dynamics
    (
        "Glaciers cover approximately ten percent of Earth's land surface and hold "
        "about 69 percent of the world's fresh water. The Antarctic Ice Sheet alone "
        "contains enough ice to raise global sea levels by roughly 58 meters if it "
        "were to melt entirely. Glaciers move under their own weight through a "
        "combination of internal deformation and basal sliding, the latter occurring "
        "when meltwater lubricates the interface between ice and bedrock. Some outlet "
        "glaciers in Greenland flow at speeds exceeding ten kilometers per year, "
        "calving massive icebergs into the North Atlantic. Glacial retreat has "
        "accelerated dramatically since the 1990s; between 2000 and 2019, the "
        "world's glaciers lost an average of 267 billion tonnes of ice annually. "
        "This meltwater contributes to sea-level rise, alters ocean salinity and "
        "circulation patterns, and threatens freshwater supplies for hundreds of "
        "millions of people who depend on glacial runoff."
    ),
    # 9 — Silk Road trade
    (
        "The Silk Road was a network of trade routes connecting East Asia to the "
        "Mediterranean, active from roughly the second century BC to the fifteenth "
        "century AD. Contrary to popular imagination, no single merchant traveled "
        "the entire route; instead, goods passed through a series of intermediaries "
        "across Central Asian oases. Silk, porcelain, and spices moved westward, "
        "while glassware, wool, and precious metals flowed east. Beyond material "
        "goods, the Silk Road transmitted religions, languages, technologies, and "
        "diseases. Buddhism spread from India to China along these routes, while "
        "papermaking and gunpowder eventually reached Europe. The Black Death of "
        "the fourteenth century also traveled the Silk Road, carried by fleas on "
        "marmots and rats. At its peak, the network spanned over 6,400 kilometers "
        "from Chang'an in China to Constantinople, passing through deserts, mountain "
        "ranges, and steppes."
    ),
    # 10 — Semiconductor fabrication
    (
        "Modern semiconductor fabrication is among the most complex manufacturing "
        "processes ever devised. A single microprocessor contains billions of "
        "transistors, each smaller than a virus, etched into silicon wafers using "
        "extreme ultraviolet lithography. The light source for EUV lithography "
        "operates at a wavelength of 13.5 nanometers, produced by blasting tiny "
        "droplets of molten tin with a powerful laser fifty thousand times per "
        "second. Each wafer undergoes hundreds of processing steps over several "
        "weeks, including deposition, etching, ion implantation, and chemical "
        "mechanical polishing. The cleanrooms where fabrication occurs are ten "
        "thousand times cleaner than a hospital operating theater, with fewer than "
        "ten particles larger than 0.1 micrometers per cubic meter of air. A "
        "single leading-edge fab costs upward of twenty billion dollars to build "
        "and equip, making semiconductor manufacturing one of the most capital- "
        "intensive industries on the planet."
    ),
    # 11 — Pollination biology
    (
        "Pollination is essential for the reproduction of roughly 87 percent of "
        "flowering plant species and underpins an estimated 35 percent of global "
        "food production. Bees are the most important pollinators, but butterflies, "
        "moths, flies, beetles, birds, and even bats also play significant roles. "
        "A single honeybee colony can visit millions of flowers per day, and the "
        "economic value of pollination services has been estimated at hundreds of "
        "billions of dollars annually. Some plants have evolved extraordinarily "
        "specific relationships with their pollinators: the fig wasp, for instance, "
        "can only reproduce inside fig fruit, while the fig depends entirely on the "
        "wasp for pollination. Colony collapse disorder, first reported in 2006, "
        "drew global attention to the decline of managed honeybee populations. "
        "Causes remain debated but likely include a combination of pesticide "
        "exposure, parasites such as Varroa destructor, habitat loss, and disease."
    ),
    # 12 — Architecture of Gothic cathedrals
    (
        "Gothic cathedrals represent one of the most ambitious building programs in "
        "European history, spanning from the twelfth to the sixteenth century. The "
        "key structural innovation was the pointed arch, which distributes weight "
        "more efficiently than the rounded Romanesque arch and allows for taller, "
        "thinner walls. Flying buttresses transferred the lateral thrust of the "
        "vaulted ceiling to external supports, freeing the walls for enormous "
        "stained-glass windows that flooded interiors with colored light. Chartres "
        "Cathedral, completed around 1220, contains over 150 windows with more than "
        "four thousand individual figures. Notre-Dame de Paris took nearly two "
        "centuries to complete and survived eight hundred years before a devastating "
        "fire in 2019. The masons who built these structures worked without modern "
        "engineering tools, relying on geometric rules of thumb passed down through "
        "guild traditions."
    ),
    # 13 — Neuroscience of memory
    (
        "Human memory is not a single system but a collection of interrelated "
        "processes distributed across multiple brain regions. Short-term memory, "
        "which holds information for seconds to minutes, relies heavily on the "
        "prefrontal cortex. Long-term memory consolidation involves the hippocampus, "
        "a seahorse-shaped structure deep in the temporal lobe. During sleep, "
        "especially during slow-wave stages, the hippocampus replays the day's "
        "experiences and gradually transfers them to the neocortex for permanent "
        "storage. This process can take weeks or even years. Emotional memories "
        "are processed in part by the amygdala, which explains why traumatic events "
        "are often remembered with unusual vividness. The discovery of long-term "
        "potentiation in 1973 provided a cellular mechanism for learning: repeated "
        "stimulation of a synapse strengthens the connection between neurons, "
        "effectively encoding information in the structure of the brain itself."
    ),
    # 14 — Fermentation and food science
    (
        "Fermentation is one of the oldest food-preservation techniques known to "
        "humanity, predating written history by thousands of years. The process "
        "relies on microorganisms, primarily yeasts and lactic acid bacteria, that "
        "convert sugars into alcohol, acids, or gases under anaerobic conditions. "
        "Bread, beer, wine, yogurt, cheese, kimchi, sauerkraut, and miso all owe "
        "their existence to fermentation. Louis Pasteur established in the 1850s "
        "that fermentation is a biological process driven by living organisms, "
        "overturning the prevailing chemical theory. In winemaking, Saccharomyces "
        "cerevisiae converts grape sugars into ethanol and carbon dioxide; the "
        "specific strain and conditions determine the wine's character. Modern "
        "fermentation science has expanded far beyond food, enabling the production "
        "of antibiotics, biofuels, and recombinant proteins. Industrial bioreactors "
        "can now culture microorganisms at scales of hundreds of thousands of liters."
    ),
    # 15 — Cartography
    (
        "The history of cartography reflects humanity's evolving understanding of "
        "the world. The oldest known maps, etched on clay tablets in Mesopotamia "
        "around 2300 BC, depicted local land features and irrigation canals. "
        "Ptolemy's Geography, written in the second century AD, introduced a "
        "coordinate system of latitude and longitude that remained influential for "
        "over a thousand years. The Age of Exploration produced increasingly "
        "accurate world maps, though distortion remained a fundamental challenge. "
        "The Mercator projection, introduced in 1569, preserved straight-line "
        "compass bearings for navigation but drastically exaggerated the size of "
        "regions near the poles. Greenland, for instance, appears similar in size "
        "to Africa on a Mercator map, despite being fourteen times smaller. Modern "
        "geographic information systems combine satellite imagery, GPS data, and "
        "computational geometry to produce maps of extraordinary precision, updated "
        "in near real time."
    ),
    # 16 — Philosophy of language
    (
        "The philosophy of language examines the nature of meaning, reference, and "
        "communication. Ludwig Wittgenstein's early work, the Tractatus Logico- "
        "Philosophicus, proposed that language mirrors the logical structure of "
        "reality, with each meaningful proposition corresponding to a possible state "
        "of affairs. He later abandoned this picture theory in the Philosophical "
        "Investigations, arguing instead that meaning arises from use within "
        "language games embedded in forms of life. Ferdinand de Saussure, the "
        "founder of structural linguistics, distinguished between langue, the "
        "abstract system of a language, and parole, its concrete use in speech. "
        "Noam Chomsky revolutionized the field in 1957 by proposing that humans "
        "possess an innate universal grammar, a biological faculty that enables "
        "language acquisition. The debate between nativist and empiricist accounts "
        "of language learning continues to shape research in linguistics, cognitive "
        "science, and artificial intelligence."
    ),
    # 17 — Volcanic activity
    (
        "Volcanic eruptions are among the most powerful natural phenomena on Earth, "
        "capable of reshaping landscapes and altering global climate. The 1815 "
        "eruption of Mount Tambora in Indonesia ejected an estimated 160 cubic "
        "kilometers of material into the atmosphere, causing the Year Without a "
        "Summer in 1816. Crops failed across Europe and North America, leading to "
        "widespread famine and social unrest. Supervolcanoes, such as the caldera "
        "beneath Yellowstone National Park, have the potential for eruptions "
        "thousands of times larger than typical volcanic events. The last Yellowstone "
        "supereruption occurred roughly 640,000 years ago and blanketed much of "
        "North America in ash. Despite their destructive potential, volcanoes also "
        "create fertile soils, geothermal energy sources, and new land. Iceland, "
        "situated on the Mid-Atlantic Ridge, generates nearly all its electricity "
        "from geothermal and hydroelectric power, both linked to volcanic geology."
    ),
    # 18 — Typography and design
    (
        "Typography, the art and technique of arranging type, influences how we "
        "perceive and process written information. The earliest typefaces, designed "
        "for Gutenberg's press, imitated the handwritten letterforms of medieval "
        "scribes. In the eighteenth century, designers like John Baskerville and "
        "Giambattista Bodoni introduced typefaces with higher contrast between thick "
        "and thin strokes, reflecting Enlightenment ideals of clarity and reason. "
        "The sans-serif typeface emerged in the nineteenth century and became "
        "synonymous with modernism; Helvetica, designed in 1957 by Max Miedinger, "
        "remains one of the most widely used typefaces in the world. Digital "
        "typography introduced variable fonts, which encode an entire family of "
        "weights and widths in a single file. Research consistently shows that "
        "typography affects reading speed, comprehension, and emotional response, "
        "making it a critical but often overlooked element of effective communication."
    ),
    # 19 — Cryptography
    (
        "Cryptography has evolved from simple substitution ciphers used by Julius "
        "Caesar to the complex mathematical algorithms that secure modern digital "
        "communication. The Caesar cipher, which shifts each letter by a fixed "
        "number of positions, is trivially broken by frequency analysis. The Enigma "
        "machine, used by Nazi Germany during World War II, employed a series of "
        "rotating electromechanical rotors to produce polyalphabetic substitutions "
        "that were considered unbreakable. Alan Turing and his colleagues at "
        "Bletchley Park cracked Enigma using a combination of mathematical insight "
        "and electromechanical computers called bombes. The invention of public-key "
        "cryptography in the 1970s by Diffie, Hellman, and later Rivest, Shamir, "
        "and Adleman transformed the field, enabling secure communication between "
        "parties who have never met. Today, RSA and elliptic curve cryptography "
        "protect trillions of dollars in daily financial transactions."
    ),
    # 20 — Soil science
    (
        "Soil is a complex living system composed of mineral particles, organic "
        "matter, water, air, and a staggering diversity of organisms. A single "
        "teaspoon of healthy topsoil can contain more microorganisms than there are "
        "people on Earth. Soil formation is extraordinarily slow: it takes roughly "
        "five hundred to a thousand years to produce one inch of topsoil through "
        "the weathering of rock, accumulation of organic material, and biological "
        "activity. The world's soils store more carbon than the atmosphere and all "
        "living plants combined, making them a critical component of the global "
        "carbon cycle. Agricultural practices such as deep plowing, monoculture, "
        "and excessive fertilizer use can degrade soil structure and deplete "
        "nutrients. Regenerative agriculture, which emphasizes cover cropping, "
        "minimal tillage, and crop rotation, aims to restore soil health while "
        "maintaining productivity. Healthy soils also improve water retention and "
        "reduce the risk of flooding and erosion."
    ),
    # 21 — Antarctic exploration
    (
        "The race to the South Pole in the early twentieth century remains one of "
        "the most dramatic episodes in the history of exploration. Roald Amundsen's "
        "Norwegian team reached the pole on December 14, 1911, using dog sleds and "
        "carefully pre-positioned supply depots. Robert Falcon Scott's British "
        "expedition arrived 34 days later, only to find Amundsen's tent and flag "
        "already in place. Scott and his four companions perished on the return "
        "journey, trapped by a blizzard just 11 miles from a supply depot. Ernest "
        "Shackleton's Endurance expedition of 1914 to 1917, though it never reached "
        "the continent, became legendary for the crew's survival after their ship "
        "was crushed by pack ice in the Weddell Sea. Today, Antarctica hosts over "
        "seventy research stations operated by thirty countries, studying climate, "
        "glaciology, astronomy, and biology in one of the most extreme environments "
        "on the planet."
    ),
    # 22 — Probability theory
    (
        "Probability theory, the mathematical framework for reasoning about "
        "uncertainty, traces its origins to a 1654 correspondence between Blaise "
        "Pascal and Pierre de Fermat concerning the fair division of stakes in an "
        "interrupted game of chance. Andrey Kolmogorov formalized the theory in "
        "1933 with his axioms, establishing probability as a branch of measure "
        "theory. The central limit theorem, one of the most important results in "
        "statistics, states that the sum of a large number of independent random "
        "variables tends toward a normal distribution regardless of the individual "
        "distributions. Bayesian inference, named after Thomas Bayes, provides a "
        "principled method for updating beliefs in light of new evidence. Monte "
        "Carlo methods, developed during the Manhattan Project, use random sampling "
        "to approximate solutions to deterministic problems that are analytically "
        "intractable. These tools are now indispensable in fields ranging from "
        "physics and finance to machine learning and drug discovery."
    ),
    # 23 — Ancient navigation
    (
        "Polynesian navigators crossed thousands of miles of open Pacific Ocean "
        "using techniques developed over millennia without any written instruments. "
        "They read the stars, tracking the rising and setting points of specific "
        "constellations to maintain course through the night. During the day, they "
        "observed the direction of ocean swells, the behavior of seabirds, and the "
        "color and temperature of the water to detect the proximity of land. "
        "Stick charts, constructed from palm ribs and cowrie shells, encoded "
        "patterns of ocean currents and wave refraction around islands. These "
        "methods enabled the settlement of Hawaii, New Zealand, and Easter Island, "
        "some of the most remote inhabited places on Earth. The voyaging canoe "
        "Hokule'a, launched in 1975, demonstrated the viability of traditional "
        "navigation by sailing from Hawaii to Tahiti without modern instruments, "
        "sparking a cultural renaissance across the Pacific Islands."
    ),
    # 24 — Industrial Revolution
    (
        "The Industrial Revolution, beginning in Britain in the mid-eighteenth "
        "century, fundamentally transformed the global economy and human society. "
        "James Watt's improved steam engine, patented in 1769, provided a reliable "
        "source of mechanical power that freed industry from dependence on water "
        "mills and animal labor. Cotton textile production mechanized rapidly, with "
        "spinning jennies, water frames, and power looms increasing output by orders "
        "of magnitude. Urbanization accelerated as workers migrated from farms to "
        "factory towns, creating unprecedented concentrations of population and "
        "pollution. By 1850, Britain produced more than half the world's coal and "
        "manufactured goods. The social consequences were profound: child labor, "
        "dangerous working conditions, and vast wealth inequality fueled reform "
        "movements and eventually reshaped labor laws, education, and public health "
        "across the industrialized world."
    ),
]

# Distractor keys for multi-key mode (RULER MK-NIAH)
DISTRACTOR_KEYS = [
    "The secret password is",
    "The hidden code is",
    "The encrypted token is",
    "The classified number is",
    "The private identifier is",
    "The concealed value is",
    "The restricted sequence is",
    "The confidential digit is",
    "The protected cipher is",
    "The obscured key is",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Needle:
    """A single needle embedded in the haystack."""
    key: str          # e.g., "The special magic number is"
    value: str        # e.g., "7281937"
    depth_pct: float  # 0.0-1.0, where in the haystack it's inserted

    @property
    def sentence(self) -> str:
        return f"{self.key}: {self.value}."


@dataclass
class TrialResult:
    """Result of one test query."""
    expected: str
    response: str
    found: bool
    needle_depth_pct: float = 0.0
    context_length: int = 0


@dataclass
class ConfigResult:
    """Result for one test configuration."""
    mode: str
    context_length: int
    cache_type: str
    needle_depth_pct: float = 0.5  # primary depth for single mode
    needle_count: int = 1          # for multi-value mode
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.found) / len(self.trials) * 100

    @property
    def passed(self) -> bool:
        return all(t.found for t in self.trials)


# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------

def _make_magic_number(rng: random.Random) -> str:
    """Generate a 7-digit random number as a string."""
    return str(rng.randint(1000000, 9999999))


def _build_filler(target_chars: int, rng: random.Random) -> list[str]:
    """Build a list of filler paragraphs totaling approximately target_chars.

    Paragraphs are shuffled and NOT repeated within a single test to avoid
    giving the model positional shortcuts via repeated text.
    """
    paragraphs = list(FILLER_PARAGRAPHS)
    rng.shuffle(paragraphs)

    result: list[str] = []
    total = 0
    idx = 0
    while total < target_chars:
        para = paragraphs[idx % len(paragraphs)]
        # After exhausting unique paragraphs, we continue cycling but this
        # only happens for very large contexts (>~100K chars).
        result.append(para)
        total += len(para) + 2  # +2 for "\n\n" separator
        idx += 1
    return result


def _insert_needles_into_paragraphs(
    paragraphs: list[str],
    needles: list[Needle],
) -> str:
    """Insert needles at their specified depth positions within the paragraph list.

    Depth 0% = before the first paragraph, 100% = after the last paragraph.
    """
    n = len(paragraphs)
    # Build list of (insert_index, needle_sentence) sorted by depth
    insertions: list[tuple[int, str]] = []
    for needle in needles:
        # Map depth_pct to an index in [0, n]
        idx = int(needle.depth_pct * n)
        idx = max(0, min(idx, n))
        insertions.append((idx, needle.sentence))

    # Insert in reverse order so earlier insertions don't shift indices
    insertions.sort(key=lambda x: x[0], reverse=True)
    for idx, sentence in insertions:
        paragraphs.insert(idx, sentence)

    return "\n\n".join(paragraphs)


def generate_haystack_single(
    needle: Needle,
    target_chars: int,
    rng: random.Random,
) -> str:
    """Generate haystack for single-needle mode."""
    paragraphs = _build_filler(target_chars, rng)
    return _insert_needles_into_paragraphs(paragraphs, [needle])


def generate_haystack_multi_key(
    real_needle: Needle,
    distractors: list[Needle],
    target_chars: int,
    rng: random.Random,
) -> str:
    """Generate haystack for multi-key mode (real needle + distractors)."""
    all_needles = [real_needle] + distractors
    paragraphs = _build_filler(target_chars, rng)
    return _insert_needles_into_paragraphs(paragraphs, all_needles)


def generate_haystack_multi_value(
    needles: list[Needle],
    target_chars: int,
    rng: random.Random,
) -> str:
    """Generate haystack for multi-value mode (same key, multiple values)."""
    paragraphs = _build_filler(target_chars, rng)
    return _insert_needles_into_paragraphs(paragraphs, needles)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_active_server: Optional[subprocess.Popen] = None
_server_stderr_file: Optional[str] = None


def _cleanup_server() -> None:
    """Kill the server process if still running (atexit / signal handler)."""
    global _active_server, _server_stderr_file
    if _active_server is not None:
        try:
            _active_server.terminate()
            _active_server.wait(timeout=5)
        except Exception:
            try:
                _active_server.kill()
            except Exception:
                pass
        _active_server = None
    # Clean up stderr temp file if it exists
    if _server_stderr_file is not None:
        try:
            os.unlink(_server_stderr_file)
        except OSError:
            pass
        _server_stderr_file = None


atexit.register(_cleanup_server)


def _signal_handler(signum: int, frame) -> None:
    _cleanup_server()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _find_free_port(start: int = 8090) -> int:
    """Find a free port starting from `start`.

    NOTE: There is an inherent TOCTOU race here -- the port could be grabbed by
    another process between our check and llama-server's bind.  Fixing this
    properly would require llama-server to support port 0 (OS-assigned) and
    report the actual port back, which it currently does not.
    """
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 99}")


def start_server(
    llama_dir: Path,
    model_path: Path,
    cache_type: str,
    context_size: int,
    port: int,
    verbose: bool = False,
    server_timeout: int = 120,
    server_bin_override: Optional[Path] = None,
) -> subprocess.Popen:
    """Start llama-server and wait until it's healthy."""
    global _active_server, _server_stderr_file

    if server_bin_override is not None:
        server_bin = server_bin_override
    else:
        server_bin = llama_dir / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        raise FileNotFoundError(f"llama-server not found at {server_bin}")

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "-c", str(context_size),
        "-ngl", "99",
        "-fa", "on",
        "--port", str(port),
        "-np", "1",
        "--jinja",
    ]

    if verbose:
        print(f"  [CMD] {' '.join(cmd)}")

    # Always capture stderr to a temp file for debugging, even in non-verbose mode.
    if verbose:
        stderr_dest = None  # inherit terminal
        stdout_dest = None
        _server_stderr_file = None
    else:
        stderr_tmp = tempfile.NamedTemporaryFile(
            prefix="niah_server_stderr_", suffix=".log", delete=False, mode="w"
        )
        _server_stderr_file = stderr_tmp.name
        stderr_dest = stderr_tmp
        stdout_dest = subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_dest,
        stderr=stderr_dest,
    )
    _active_server = proc

    # Poll /health until ready
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + server_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_snippet = _read_server_stderr()
            raise RuntimeError(
                f"llama-server exited prematurely with code {proc.returncode}\n"
                f"Server stderr:\n{stderr_snippet}"
            )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return proc
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.5)

    stderr_snippet = _read_server_stderr()
    proc.terminate()
    proc.wait(timeout=5)
    _active_server = None
    raise TimeoutError(
        f"llama-server did not become healthy within {server_timeout} seconds\n"
        f"Server stderr:\n{stderr_snippet}"
    )


def _read_server_stderr(max_bytes: int = 4096) -> str:
    """Read the captured server stderr log, if available."""
    if _server_stderr_file is None:
        return "(stderr not captured -- run with --verbose to see server output)"
    try:
        with open(_server_stderr_file, "r") as f:
            content = f.read()
        # Return the last max_bytes chars (tail) which are most useful
        if len(content) > max_bytes:
            return f"... (truncated)\n{content[-max_bytes:]}"
        return content if content else "(empty)"
    except OSError:
        return "(could not read stderr log)"


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server."""
    global _active_server, _server_stderr_file
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
    _active_server = None
    # Clean up stderr temp file
    if _server_stderr_file is not None:
        try:
            os.unlink(_server_stderr_file)
        except OSError:
            pass
        _server_stderr_file = None


# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------

def _query_server(
    port: int,
    user_content: str,
    timeout: int = 300,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request and return the content field.

    Scores from content field ONLY (not reasoning_content) — Qwen3.5 thinking
    mode fix. The reasoning chain may mention other needles' values.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = json.dumps({
        "model": "model",
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "seed": SEED,
        # Qwen3.5 uses thinking mode — needs enough tokens to finish thinking + answer.
        "max_tokens": 2048,
    }).encode()

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if content:
                    # Strip any remaining thinking tags
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return content
                return ""
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Failed to query server after {max_retries} attempts: {e}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unexpected response format: {e}")

    return ""  # unreachable, but keeps mypy happy


def _score_single(response: str, expected: str) -> bool:
    """Binary scoring: extract first 7-digit number from response, exact match."""
    match = re.search(r'\b(\d{7})\b', response)
    return match is not None and match.group(1) == expected


def _score_multi_value(response: str, expected_values: list[str]) -> list[bool]:
    """Score multi-value: check which expected values appear in the response."""
    found_numbers = re.findall(r'\b(\d{7})\b', response)
    return [val in found_numbers for val in expected_values]


# ---------------------------------------------------------------------------
# Test runners by mode
# ---------------------------------------------------------------------------

def _run_single_trial(
    port: int,
    haystack: str,
    needle: Needle,
    timeout: int,
    verbose: bool,
) -> TrialResult:
    """Run one single-needle trial."""
    user_content = (
        f"{haystack}\n\n"
        "What is the special magic number mentioned in the above text? "
        "Reply with ONLY the number, nothing else."
    )

    response = _query_server(port, user_content, timeout=timeout)
    found = _score_single(response, needle.value)

    if verbose:
        status = "HIT" if found else "MISS"
        print(f"    {status} (expected={needle.value}, got={response!r})")

    return TrialResult(
        expected=needle.value,
        response=response,
        found=found,
        needle_depth_pct=needle.depth_pct,
    )


def run_single_mode(
    args: argparse.Namespace,
    llama_dir: Path,
    model_path: Path,
) -> list[ConfigResult]:
    """Kamradt single-needle: sweep depth (0-100%) x context length."""
    depths_sweep = [int(d) for d in args.depths_sweep.split(",")]
    context_lengths = [int(d) for d in args.depths.split(",")]
    cache_types = [c.strip() for c in args.cache_types.split(",")]
    port = int(args.port)
    server_bin = Path(args.server_bin) if args.server_bin else None

    total = len(depths_sweep) * len(context_lengths) * len(cache_types)
    print(f"\nSingle-needle mode: {total} configurations")
    print(f"  Depth sweep: {depths_sweep}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Cache types: {cache_types}")

    results: list[ConfigResult] = []
    config_num = 0

    for cache_type in cache_types:
        for ctx_len in context_lengths:
            # Start server once per (cache_type, ctx_len) — reuse for all depths
            server_ctx = ctx_len + 4096  # headroom for query + response
            actual_port = _find_free_port(port)
            print(f"\n  Starting server: cache={cache_type}, ctx={server_ctx}")
            proc = start_server(
                llama_dir, model_path, cache_type, server_ctx, actual_port,
                args.verbose, server_timeout=args.server_timeout,
                server_bin_override=server_bin,
            )

            try:
                for depth_pct in depths_sweep:
                    config_num += 1
                    rng = random.Random(SEED + depth_pct + ctx_len)

                    needle = Needle(
                        key="The special magic number is",
                        value=_make_magic_number(rng),
                        depth_pct=depth_pct / 100.0,
                    )

                    target_chars = int(ctx_len * args.chars_per_token)
                    haystack = generate_haystack_single(needle, target_chars, rng)

                    ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
                    print(f"    [{config_num}/{total}] cache={cache_type} "
                          f"ctx={ctx_label} depth={depth_pct}%", end=" ", flush=True)

                    trial = _run_single_trial(
                        actual_port, haystack, needle, args.query_timeout, args.verbose,
                    )
                    trial.context_length = ctx_len

                    result = ConfigResult(
                        mode="single",
                        context_length=ctx_len,
                        cache_type=cache_type,
                        needle_depth_pct=depth_pct / 100.0,
                    )
                    result.trials.append(trial)
                    results.append(result)

                    status = "✅" if trial.found else "❌"
                    if not args.verbose:
                        print(status)
            finally:
                stop_server(proc)

    return results


def run_multi_key_mode(
    args: argparse.Namespace,
    llama_dir: Path,
    model_path: Path,
) -> list[ConfigResult]:
    """RULER MK-NIAH: real needle + distractors, sweep context length."""
    context_lengths = [int(d) for d in args.depths.split(",")]
    cache_types = [c.strip() for c in args.cache_types.split(",")]
    num_distractors = args.num_distractors
    port = int(args.port)
    server_bin = Path(args.server_bin) if args.server_bin else None

    total = len(context_lengths) * len(cache_types)
    print(f"\nMulti-key mode: {total} configurations, {num_distractors} distractors each")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Cache types: {cache_types}")

    results: list[ConfigResult] = []
    config_num = 0

    for cache_type in cache_types:
        for ctx_len in context_lengths:
            config_num += 1
            rng = random.Random(SEED + ctx_len)

            # Real needle at 50% depth
            real_needle = Needle(
                key="The special magic number is",
                value=_make_magic_number(rng),
                depth_pct=0.5,
            )

            # Distractor needles spread evenly
            distractors: list[Needle] = []
            for i in range(num_distractors):
                d_key = DISTRACTOR_KEYS[i % len(DISTRACTOR_KEYS)]
                d_pos = (i + 1) / (num_distractors + 2)  # spread across 0-1
                distractors.append(Needle(
                    key=d_key,
                    value=_make_magic_number(rng),
                    depth_pct=d_pos,
                ))

            target_chars = int(ctx_len * args.chars_per_token)
            haystack = generate_haystack_multi_key(real_needle, distractors, target_chars, rng)

            server_ctx = ctx_len + 4096
            actual_port = _find_free_port(port)
            ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
            print(f"\n  [{config_num}/{total}] cache={cache_type} ctx={ctx_label}", end=" ", flush=True)

            proc = start_server(
                llama_dir, model_path, cache_type, server_ctx, actual_port,
                args.verbose, server_timeout=args.server_timeout,
                server_bin_override=server_bin,
            )

            try:
                user_content = (
                    f"{haystack}\n\n"
                    "What is the special magic number mentioned in the above text? "
                    "Reply with ONLY the number, nothing else."
                )
                response = _query_server(actual_port, user_content, timeout=args.query_timeout)
                found = _score_single(response, real_needle.value)

                trial = TrialResult(
                    expected=real_needle.value,
                    response=response,
                    found=found,
                    needle_depth_pct=0.5,
                    context_length=ctx_len,
                )

                result = ConfigResult(
                    mode="multi-key",
                    context_length=ctx_len,
                    cache_type=cache_type,
                    needle_depth_pct=0.5,
                    needle_count=1 + num_distractors,
                )
                result.trials.append(trial)
                results.append(result)

                status = "✅" if found else "❌"
                print(status)
                if args.verbose:
                    print(f"    expected={real_needle.value}, got={response!r}")
            finally:
                stop_server(proc)

    return results


def run_multi_value_mode(
    args: argparse.Namespace,
    llama_dir: Path,
    model_path: Path,
) -> list[ConfigResult]:
    """RULER MV-NIAH: multiple same-key needles, sweep length x count."""
    context_lengths = [int(d) for d in args.depths.split(",")]
    cache_types = [c.strip() for c in args.cache_types.split(",")]
    value_counts = [int(n) for n in args.value_counts.split(",")]
    port = int(args.port)
    server_bin = Path(args.server_bin) if args.server_bin else None

    total = len(context_lengths) * len(cache_types) * len(value_counts)
    print(f"\nMulti-value mode: {total} configurations")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Value counts: {value_counts}")
    print(f"  Cache types: {cache_types}")

    results: list[ConfigResult] = []
    config_num = 0

    for cache_type in cache_types:
        for ctx_len in context_lengths:
            server_ctx = ctx_len + 4096
            actual_port = _find_free_port(port)
            print(f"\n  Starting server: cache={cache_type}, ctx={server_ctx}")
            proc = start_server(
                llama_dir, model_path, cache_type, server_ctx, actual_port,
                args.verbose, server_timeout=args.server_timeout,
                server_bin_override=server_bin,
            )

            try:
                for vc in value_counts:
                    config_num += 1
                    rng = random.Random(SEED + ctx_len + vc)

                    # Create vc needles with the same key at different positions
                    needles: list[Needle] = []
                    for i in range(vc):
                        pos = (i + 1) / (vc + 1)  # spread evenly
                        needles.append(Needle(
                            key="The special magic number is",
                            value=_make_magic_number(rng),
                            depth_pct=pos,
                        ))

                    target_chars = int(ctx_len * args.chars_per_token)
                    haystack = generate_haystack_multi_value(needles, target_chars, rng)

                    ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
                    print(f"    [{config_num}/{total}] cache={cache_type} "
                          f"ctx={ctx_label} values={vc}", end=" ", flush=True)

                    user_content = (
                        f"{haystack}\n\n"
                        "What are ALL the special magic numbers mentioned in the above text? "
                        "Reply with ONLY the numbers separated by commas, nothing else."
                    )
                    response = _query_server(actual_port, user_content, timeout=args.query_timeout)
                    expected_values = [n.value for n in needles]
                    hits = _score_multi_value(response, expected_values)

                    result = ConfigResult(
                        mode="multi-value",
                        context_length=ctx_len,
                        cache_type=cache_type,
                        needle_count=vc,
                    )

                    for needle, hit in zip(needles, hits):
                        result.trials.append(TrialResult(
                            expected=needle.value,
                            response=response,
                            found=hit,
                            needle_depth_pct=needle.depth_pct,
                            context_length=ctx_len,
                        ))
                    results.append(result)

                    hit_count = sum(hits)
                    status = "✅" if all(hits) else f"⚠️ {hit_count}/{vc}"
                    print(status)
                    if args.verbose:
                        print(f"    expected={expected_values}, got={response!r}")
            finally:
                stop_server(proc)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _build_heatmap_table(
    results: list[ConfigResult],
    cache_type: str,
    model_name: str,
) -> str:
    """Build a 2D heatmap table (depth rows x length columns) for single mode."""
    # Filter results for this cache type
    ct_results = [r for r in results if r.cache_type == cache_type]
    if not ct_results:
        return f"## Single Needle Retrieval: {cache_type}\n\n(no results)\n"

    depths = sorted(set(int(r.needle_depth_pct * 100) for r in ct_results))
    lengths = sorted(set(r.context_length for r in ct_results))

    # Build lookup: (depth_pct_int, ctx_len) -> passed
    lookup: dict[tuple[int, int], Optional[bool]] = {}
    for r in ct_results:
        d = int(r.needle_depth_pct * 100)
        lookup[(d, r.context_length)] = r.passed

    # Header
    len_labels = []
    for l in lengths:
        len_labels.append(f"{l // 1024}K" if l >= 1024 else str(l))

    header = "| Depth  |"
    sep = "|--------|"
    for label in len_labels:
        header += f" {label:<5}|"
        sep += f"------|"

    lines = [
        f"## Single Needle Retrieval: {cache_type}",
        "",
        header,
        sep,
    ]

    for d in depths:
        row = f"| {d:<4}%  |"
        for l in lengths:
            val = lookup.get((d, l))
            if val is None:
                cell = " ERR "
            elif val:
                cell = " ✅   "
            else:
                cell = " ❌   "
            row += f"{cell}|"
        lines.append(row)

    return "\n".join(lines)


def _build_delta_table(
    results: list[ConfigResult],
    baseline: str,
    compare: str,
) -> str:
    """Build a delta table showing where compare differs from baseline."""
    base_results = {
        (int(r.needle_depth_pct * 100), r.context_length): r.passed
        for r in results if r.cache_type == baseline
    }
    comp_results = {
        (int(r.needle_depth_pct * 100), r.context_length): r.passed
        for r in results if r.cache_type == compare
    }

    if not base_results or not comp_results:
        return ""

    depths = sorted(set(k[0] for k in base_results))
    lengths = sorted(set(k[1] for k in base_results))

    len_labels = [f"{l // 1024}K" if l >= 1024 else str(l) for l in lengths]

    header = "| Depth  |"
    sep = "|--------|"
    for label in len_labels:
        header += f" {label:<5}|"
        sep += f"------|"

    lines = [
        f"## Delta: {compare} vs {baseline}",
        "",
        header,
        sep,
    ]

    has_diff = False
    for d in depths:
        row = f"| {d:<4}%  |"
        for l in lengths:
            b = base_results.get((d, l))
            c = comp_results.get((d, l))
            if b is None or c is None:
                cell = " N/A "
            elif b == c:
                cell = "  =  "
            elif c and not b:
                cell = " +🟢 "
                has_diff = True
            else:
                cell = " -🔴 "
                has_diff = True
            row += f"{cell}|"
        lines.append(row)

    if not has_diff:
        lines.append("")
        lines.append("No differences detected.")

    return "\n".join(lines)


def _build_multi_key_table(
    results: list[ConfigResult],
    model_name: str,
) -> str:
    """Build table for multi-key mode results."""
    cache_types = sorted(set(r.cache_type for r in results))
    lengths = sorted(set(r.context_length for r in results))

    len_labels = [f"{l // 1024}K" if l >= 1024 else str(l) for l in lengths]

    header = "| Cache Type |"
    sep = "|------------|"
    for label in len_labels:
        header += f" {label:<5}|"
        sep += f"------|"

    lines = [
        f"## Multi-Key Retrieval (MK-NIAH)",
        "",
        header,
        sep,
    ]

    lookup: dict[tuple[str, int], bool] = {}
    for r in results:
        lookup[(r.cache_type, r.context_length)] = r.passed

    for ct in cache_types:
        row = f"| {ct:<10} |"
        for l in lengths:
            val = lookup.get((ct, l))
            if val is None:
                cell = " ERR "
            elif val:
                cell = " ✅   "
            else:
                cell = " ❌   "
            row += f"{cell}|"
        lines.append(row)

    return "\n".join(lines)


def _build_multi_value_table(
    results: list[ConfigResult],
    model_name: str,
) -> str:
    """Build table for multi-value mode results."""
    cache_types = sorted(set(r.cache_type for r in results))
    lengths = sorted(set(r.context_length for r in results))
    value_counts = sorted(set(r.needle_count for r in results))

    lines = [f"## Multi-Value Retrieval (MV-NIAH)", ""]

    for ct in cache_types:
        len_labels = [f"{l // 1024}K" if l >= 1024 else str(l) for l in lengths]

        header = "| Values |"
        sep = "|--------|"
        for label in len_labels:
            header += f" {label:<7}|"
            sep += f"--------|"

        lines.extend([f"### {ct}", "", header, sep])

        lookup: dict[tuple[int, int], float] = {}
        for r in results:
            if r.cache_type == ct:
                lookup[(r.needle_count, r.context_length)] = r.accuracy_pct

        for vc in value_counts:
            row = f"| {vc:<6} |"
            for l in lengths:
                pct = lookup.get((vc, l))
                if pct is None:
                    cell = " ERR    "
                elif pct == 100.0:
                    cell = " ✅ 100%"
                elif pct == 0.0:
                    cell = " ❌   0%"
                else:
                    cell = f" ⚠️ {pct:3.0f}%"
                row += f"{cell}|"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def build_output(results: list[ConfigResult], model_name: str, mode: str) -> str:
    """Build the full markdown output for the given mode."""
    parts = [
        f"# TurboQuant NIAH Benchmark: {model_name}",
        f"Mode: {mode} | Seed: {SEED} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if mode == "single":
        cache_types = sorted(set(r.cache_type for r in results))
        for ct in cache_types:
            parts.append(_build_heatmap_table(results, ct, model_name))
            parts.append("")

        # Delta table if we have exactly 2 cache types
        if len(cache_types) == 2:
            parts.append(_build_delta_table(results, cache_types[0], cache_types[1]))
            parts.append("")

    elif mode == "multi-key":
        parts.append(_build_multi_key_table(results, model_name))
        parts.append("")

    elif mode == "multi-value":
        parts.append(_build_multi_value_table(results, model_name))
        parts.append("")

    return "\n".join(parts)


def save_results(
    results: list[ConfigResult],
    model_name: str,
    mode: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save results as JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON results
    json_path = output_dir / f"niah_{mode}_{timestamp}.json"
    json_data = {
        "model": model_name,
        "mode": mode,
        "seed": SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "mode": r.mode,
                "context_length": r.context_length,
                "cache_type": r.cache_type,
                "needle_depth_pct": r.needle_depth_pct,
                "needle_count": r.needle_count,
                "accuracy_pct": r.accuracy_pct,
                "passed": r.passed,
                "trials": [
                    {
                        "expected": t.expected,
                        "response": t.response,
                        "found": t.found,
                        "needle_depth_pct": t.needle_depth_pct,
                        "context_length": t.context_length,
                    }
                    for t in r.trials
                ],
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(json_data, f, indent=2)

    # Markdown
    md_path = output_dir / f"niah_{mode}_{timestamp}.md"
    md_content = build_output(results, model_name, mode)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md_content + "\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboQuant NIAH Benchmark v2 — Kamradt + RULER methodology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes:
              single       Kamradt single-needle: sweep depth (0-100%%) x context length
              multi-key    RULER MK-NIAH: real needle + distractors, sweep context length
              multi-value  RULER MV-NIAH: multiple same-key needles, sweep length x count

            Examples:
              python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf
              python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode single
              python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode multi-key --num-distractors 5
              python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf --mode multi-value --value-counts 2,4,8
        """),
    )

    parser.add_argument(
        "llama_dir",
        help="Path to the llama.cpp directory (must contain build/bin/llama-server)",
    )
    parser.add_argument(
        "model_path",
        default=None,
        nargs="?",
        help="Path to the GGUF model file (required)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi-key", "multi-value"],
        default="single",
        help="Test mode (default: %(default)s)",
    )
    parser.add_argument(
        "--depths",
        default="4096,8192,16384,32768",
        help="Comma-separated context lengths in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--depths-sweep",
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated needle depth positions in %% for single mode (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-types",
        default="q8_0,turbo3",
        help="Comma-separated cache types to test (default: %(default)s)",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=3,
        help="Number of distractor needles for multi-key mode (default: %(default)s)",
    )
    parser.add_argument(
        "--value-counts",
        default="2,4,8",
        help="Comma-separated value counts for multi-value mode (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        default="8090",
        help="Base port for llama-server (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for result files (default: artifacts/niah/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show server output and detailed info",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for llama-server to become healthy (default: %(default)s)",
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each query request (default: %(default)s). "
             "32K prefill on M1 can take 130s+, so the default is generous.",
    )
    parser.add_argument(
        "--server-bin",
        default=None,
        help="Path to a specific llama-server binary. "
             "Overrides the default <llama_dir>/build/bin/llama-server.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Approximate characters per token for haystack sizing (default: %(default)s).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate model_path is provided
    if args.model_path is None:
        print(
            "Error: model_path is required. Pass the path to a GGUF model file as the second argument.\n"
            "Usage: python3 tools/validation/niah_test.py /path/to/llama.cpp /path/to/model.gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    llama_dir = Path(args.llama_dir)
    model_path = Path(args.model_path)

    # Validate paths
    server_bin: Optional[Path] = None
    if args.server_bin:
        server_bin = Path(args.server_bin)
        if not server_bin.exists():
            print(f"Error: server binary not found at {server_bin}", file=sys.stderr)
            sys.exit(1)
    else:
        if not llama_dir.exists():
            print(f"Error: llama.cpp directory not found: {llama_dir}", file=sys.stderr)
            sys.exit(1)
        default_bin = llama_dir / "build" / "bin" / "llama-server"
        if not default_bin.exists():
            print(f"Error: llama-server not found at {default_bin}", file=sys.stderr)
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parents[2] / "artifacts" / "niah"
        )
    output_dir = Path(args.output_dir)
    model_name = model_path.stem

    print(f"{'=' * 60}")
    print(f"  TurboQuant NIAH Benchmark v2")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {model_name}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    # Dispatch to mode-specific runner
    if args.mode == "single":
        results = run_single_mode(args, llama_dir, model_path)
    elif args.mode == "multi-key":
        results = run_multi_key_mode(args, llama_dir, model_path)
    elif args.mode == "multi-value":
        results = run_multi_value_mode(args, llama_dir, model_path)
    else:
        print(f"Error: unknown mode {args.mode}", file=sys.stderr)
        sys.exit(1)

    # Print output
    output = build_output(results, model_name, args.mode)
    print(f"\n\n{output}\n")

    # Save results
    json_path, md_path = save_results(results, model_name, args.mode, output_dir)
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Table: {md_path}")


if __name__ == "__main__":
    main()
