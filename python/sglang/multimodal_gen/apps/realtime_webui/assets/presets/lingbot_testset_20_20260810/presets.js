// Generated from manifest.jsonl by build_presets.mjs.
(function exposeLingBotTestset(root, factory) {
  const presets = factory();
  if (typeof module === "object" && module.exports) module.exports = presets;
  if (root) root.LINGBOT_TESTSET_20_20260810 = presets;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildPresets() {
  return [
  {
    "id": "lingbot-testset-20260810-case-01",
    "name": "Orchard Maintenance Worker",
    "tone": "green",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of an orchard maintenance worker. A narrow-gauge orchard railway yard has a smooth packed-earth service lane between apple rows and a small timber loading shed; dormant flatcars sit on a siding while a low stone culvert leads the route toward sunlit hills. Bare forearms and the edge of a faded olive work shirt remain visible along the lower frame. The visual style is colored-pencil and watercolor illustration. At 24 FPS, strafe right for 121 frames, then move forward for 8 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_01_lbd_835e051d1792f424495669ad.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_01",
      "category": "human_hands_visible",
      "trajectoryId": "combo_d-w_121-8_nostop",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-02",
    "name": "Rail Inspector",
    "tone": "green",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of an underground rail inspector. An abandoned slate quarry adit opens into a level service passage with compacted gravel under timber roof sets. A narrow rail groove runs along the right wall, slate spoil piles are set back from the central route, and a daylight portal is visible at the far end. Dark insulated sleeves and gloved hands remain visible along the lower frame. The visual style is realistic industrial documentary photography. At 24 FPS, move backward for 85 frames, strafe right for 8 frames, then hold a steady view for 36 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_02_lbd_06818bfd6f33ee5449441143.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_02",
      "category": "human_hands_visible",
      "trajectoryId": "combo_s-d_85-8_stop36after2",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-03",
    "name": "Station Custodian",
    "tone": "green",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of an orbital station custodian. A rotating orbital greenhouse ring presents a gently curving grated service path beneath transparent planting trays and ribbed aluminum arches; a compact filter cartridge is clipped to a service panel on the left within arm reach. White pressure-suit gloves and a blue utility tether remain visible along the lower frame. The visual style is geometric cut-paper diorama. At 24 FPS, tilt the view upward for 58 frames, then turn the view right for 71 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_03_lbd_8a9b04f8fafc479f01f6085c.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_03",
      "category": "human_hands_visible",
      "trajectoryId": "combo_i-l_58-71_nostop",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-04",
    "name": "Adult",
    "tone": "green",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a small-boat captain. Sheltered riverboat basin with broad water lane between timber pilings and brick warehouses, an open lock gate ahead. Bare forearms and a weathered wooden tiller remain visible along the lower frame. The visual style is realistic river documentary photography. At 24 FPS, turn the view left for 17 frames, hold a steady view for 36 frames, then tilt the view downward for 76 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_04_lbd_93a2a3510f059ea688c8d58f.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_04",
      "category": "human_hands_visible",
      "trajectoryId": "combo_j-k_17-76_stop36after1",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-05",
    "name": "Beginner Swimmer",
    "tone": "green",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a beginner swimmer. A tree-lined neighborhood pool has a wide tan paver corridor curving gently around a shallow teaching basin, with low stone planters on the outer side and an empty row of kickboards on a rack beside the water. Bare forearms remain visible along the lower frame. The visual style is hand-painted gouache illustration. At 24 FPS, strafe right for 121 frames, then turn the view right for 8 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_05_lbd_a14c89306ffafa3fc8b767ab.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_05",
      "category": "human_hands_visible",
      "trajectoryId": "combo_d-l_121-8_nostop",
      "actionFamily": "wasd_to_ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-06",
    "name": "Conference Attendee",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a hotel conference guest. A bright hotel luggage vestibule with a broad terrazzo aisle from revolving doors toward a frosted-glass service corridor, oak luggage shelves lining the walls. The environment fills the frame from an eye-level human viewpoint. The visual style is hand-painted gouache illustration. At 24 FPS, hold a steady view for 72 frames, then strafe left for 57 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_06_lbd_21be7d68ec8a3afc8b2f28d5.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_06",
      "category": "human_body_hidden",
      "trajectoryId": "combo_a_57_stop72beforefirst",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-07",
    "name": "Desert Field Researcher",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a desert field researcher. A dry canyon restoration wash has a level compacted-earth trail threading between low woven-brush check structures, pale sandstone walls, and a distant rain gauge mast; the corridor stays broad and clear through the center. The environment fills the frame from an eye-level human viewpoint. The visual style is naturalistic desert field photography. At 24 FPS, strafe left for 8 frames, move forward for 25 frames, then hold a steady view for 96 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_07_lbd_9f04c197a16b59f71a07f5ce.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_07",
      "category": "human_body_hidden",
      "trajectoryId": "combo_a-w_8-25_stop96after2",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-08",
    "name": "Marine Archaeologist",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a marine archaeologist. A clear shallow-water survey lane follows a pale sand ribbon between low seagrass beds and scattered rounded amphora fragments, with a square weighted survey frame visible at mid-distance. The underwater environment fills the frame from a human viewpoint. The visual style is watercolor cel illustration. At 24 FPS, turn the view left for 9 frames, tilt the view downward for 48 frames, then hold a steady view for 72 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_08_lbd_990258e4e349628c14156537.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_08",
      "category": "human_body_hidden",
      "trajectoryId": "combo_j-k_9-48_stop72after2",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-09",
    "name": "Subway Signal Technician",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a subway signal technician. An inactive subway service gallery runs beside twin rails behind a safety line, with white tile pilasters, relay cabinets, a gently curving service path, and a visible crossover portal ahead. The environment fills the frame from an eye-level human viewpoint. The visual style is hand-painted gouache illustration. At 24 FPS, move backward for 48 frames, turn the view left for 19 frames, then move forward for 62 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_09_lbd_f7bb008efd71cfa37570eadf.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_09",
      "category": "human_body_hidden",
      "trajectoryId": "combo_s-j-w_48-19-62_nostop",
      "actionFamily": "wasd_to_ijkl_to_wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-10",
    "name": "Knife Skills Student",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a knife-skills student. A knife-skills classroom has a single long maple demonstration table offset beside a centered epoxy-floor lane; magnetic utensil rails line one wall and an empty instructor platform creates a distant landmark. The environment fills the frame from an eye-level human viewpoint. The visual style is realistic educational photography. At 24 FPS, turn the view left for 93 frames, then move backward for 36 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_10_lbd_a6730bb94a8b23d8787be635.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_10",
      "category": "human_body_hidden",
      "trajectoryId": "combo_j-s_93-36_nostop",
      "actionFamily": "ijkl_to_wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-11",
    "name": "Red Deer Stag",
    "tone": "accent",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a red deer stag. A heather-covered highland saddle stretches between low granite ridges, with a firm animal track crossing a shallow peat channel on flat stones and continuing toward a lone rowan tree; broad grass strips on either side allow alternate traversal. Branched antler tips frame the upper corners and a russet muzzle remains visible at the lower edge. The visual style is layered cut-paper diorama. At 24 FPS, move forward for 8 frames, hold a steady view for 96 frames, strafe left for 17 frames, then move backward for 8 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_11_lbd_bc42d627847ec0f3709533cf.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_11",
      "category": "nonhuman_organic",
      "trajectoryId": "combo_w-a-s_8-17-8_stop96after1",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-12",
    "name": "Lynx",
    "tone": "accent",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a lynx. A mountain lodge corridor has rough pine plank flooring and wide stone-framed windows opening toward a snowfield; the straight hall ends at a closed oak door with clear floor space on both sides. Black-tufted ears frame the upper edges and pale whisker pads remain visible at the lower edge. The visual style is realistic interior animal photography. At 24 FPS, strafe left for 129 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_12_lbd_e46340c3dda3d7eb5c0855d8.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_12",
      "category": "nonhuman_organic",
      "trajectoryId": "combo_a_129_nostop",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-13",
    "name": "Dragonfly",
    "tone": "accent",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a dragonfly. A low flight corridor follows a sunlit reed-fringed pond channel, with broad lily pads below, open water through the center, cattails kept to the margins, and a pale fallen branch forming a distant orientation landmark. Transparent veined wing roots frame the lower sides and a blue-black thorax ridge remains visible at the bottom center. The visual style is layered cut-paper diorama. At 24 FPS, tilt the view downward for 69 frames, then tilt the view upward for 60 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_13_lbd_a2c297f2a3c891085faff150.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_13",
      "category": "nonhuman_organic",
      "trajectoryId": "combo_k-i_69-60_nostop",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-14",
    "name": "Crystal Shelled Tunnel Wyrm",
    "tone": "accent",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a crystal-shelled tunnel wyrm. An abandoned salt mine opens into a level haulage corridor with a centered packed-salt track, timber support frames, pale crystal seams, and a distant circular ventilation shaft. Faceted cobalt brow plates remain visible at the lower corners. The visual style is naturalistic underground photography. At 24 FPS, tilt the view upward for 45 frames, turn the view right for 13 frames, then turn the view left for 71 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_14_lbd_c6b9020b0685a183fd3391e7.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_14",
      "category": "nonhuman_organic",
      "trajectoryId": "combo_i-l-j_45-13-71_nostop",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-15",
    "name": "Working Horse",
    "tone": "accent",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a working horse. A heritage freight depot plaza begins at an iron-framed notice wall and spreads into fan-shaped cobbles, with a near route around a loading ramp, a middle route between rail setts, and a far route toward a timber depot door. Chestnut ears and a leather browband remain visible along the upper edge. The visual style is realistic large-format location photography. At 24 FPS, strafe left for 65 frames, turn the view right for 16 frames, then hold a steady view for 48 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_15_lbd_c2c4534ee9d81726bd85bcc1.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_15",
      "category": "nonhuman_organic",
      "trajectoryId": "combo_a-l_65-16_stop48after2",
      "actionFamily": "wasd_to_ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-16",
    "name": "Shallow Draft Electric Skiff",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of an electric river patrol boat. A narrow engineered river channel passes beneath a broad concrete promenade, opening into a linear waterside picnic court with stepped granite landings and a floating safety dock aligned straight ahead. A varnished bow coaming and coiled mooring line remain visible along the lower frame. The visual style is colored-pencil illustration. At 24 FPS, strafe left for 8 frames, strafe right for 49 frames, then hold a steady view for 72 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_16_lbd_53578d7f141f6f91fb541cc0.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_16",
      "category": "nonhuman_mechanical",
      "trajectoryId": "combo_a-d_8-49_stop72after2",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-17",
    "name": "Maintenance Trolley",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a rail maintenance trolley. A wide mine inspection tunnel contains a straight narrow-gauge track, gravel shoulders usable as emergency walkways, regularly spaced support ribs, bright utility lamps, and a lit junction sign shape in the distance. A small cave salamander rests on the left rail immediately ahead of the stopped trolley. A yellow dashboard, black brake lever, and front safety rail remain visible along the lower frame. The visual style is realistic industrial location photography. At 24 FPS, move forward for 80 frames, then strafe left for 49 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_17_lbd_c7889ddd5f5b9e24fa1e54ce.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_17",
      "category": "nonhuman_mechanical",
      "trajectoryId": "combo_w-a_80-49_nostop",
      "actionFamily": "wasd"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-18",
    "name": "Six Wheel Survey Rover",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a six-wheel Mars survey rover. A broad compacted regolith track leads through a shallow ochre crater between low basalt outcrops toward a pale layered escarpment, with wheel-safe terrain and a straight central approach. A dusty white instrument mast base and angular solar-panel corners remain visible along the lower frame. The visual style is screen-printed science illustration. At 24 FPS, tilt the view downward for 13 frames, hold a steady view for 96 frames, then turn the view right for 20 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_18_lbd_b7b38628bcd6ce61003b5018.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_18",
      "category": "nonhuman_mechanical",
      "trajectoryId": "combo_k-l_13-20_stop96after1",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-19",
    "name": "Small Research Submersible",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a compact research submersible. A clear temperate kelp channel opens between rock shelves, with a sandy bottom forming a continuous navigable lane, kelp canopies held above the route, survey stakes far to one side, and a natural stone arch as the forward landmark. A yellow pressure-hull nose and paired thruster guards remain visible along the lower frame. The visual style is realistic underwater expedition photography. At 24 FPS, hold a steady view for 36 frames, tilt the view downward for 8 frames, then turn the view right for 85 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_19_lbd_ccaa276fc792bbf1efd41478.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_19",
      "category": "nonhuman_mechanical",
      "trajectoryId": "combo_k-l_8-85_stop36beforefirst",
      "actionFamily": "ijkl"
    }
  },
  {
    "id": "lingbot-testset-20260810-case-20",
    "name": "Compact Litter Sweeper",
    "tone": "blue",
    "size": "1280x704",
    "fps": 24,
    "prompt": "Continuous first-person gameplay from the viewpoint of a compact street-cleaning robot. A pedestrian riverside promenade curves gently under plane trees, with pale stone paving, cast-iron benches, low parapet walls, and a glass recycling station beside the route. A rounded green brush housing and narrow intake lip remain visible along the lower frame. The visual style is realistic urban documentary photography. At 24 FPS, move backward for 53 frames, hold a steady view for 48 frames, turn the view left for 20 frames, then move forward for 8 frames. Spatial layout, lighting, viewpoint identity, and visible viewpoint anchors remain continuous throughout the sequence.",
    "referenceUrl": "./assets/presets/lingbot_testset_20_20260810/images/case_20_lbd_aef443067c9db5fa7cbcd6c1.png",
    "mime": "image/png",
    "source": "LingBot reviewed testset 20260810",
    "metadata": {
      "caseId": "case_20",
      "category": "nonhuman_mechanical",
      "trajectoryId": "combo_s-j-w_53-20-8_stop48after1",
      "actionFamily": "wasd_to_ijkl_to_wasd"
    }
  }
];
});
