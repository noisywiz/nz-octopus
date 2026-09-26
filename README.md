# nz-octopus

> **Note:** this project is written entirely by an LLM (AI-generated code);
> a human directs it and reviews the results.

A 2D aquarium where bacteria live with tabular Q-learning brains.
They are born fully random and learn for life: find food by smell,
avoid the walls, and survive hunger. Brains are saved to `brains.json` —
a colony remembers its experience across runs.

## Run

```bash
uv run main.py          # a fresh colony; brains.json is neither read nor written
uv run main.py --brain  # resume the saved brains and save on exit
uv run python -m unittest discover  # tests
```

In the canvas: `f` pour food, `+/-` speed, `space` pause, `s` save the
brains to `brains.json`, `f11` fullscreen, `q` quit.
The window is freely resizable (maximize button, `f11`): the tank itself
grows or shrinks to match at the same zoom — more screen means more water,
the dunes rebuild, and everything clamps into the new bounds.
Without `--brain` the file is written only when you press `s`; with the
flag it is also saved on exit (including Ctrl-C).

State (36 variants): smell gradient (rising/falling/flat) × side
(left/right/symmetric) × proximity (4 levels). Actions: 8 swimming
directions + rest. A creature cannot see food through the water — it only
smells it.

Aging: past hunger 75 a creature grays out and slows down (down to 40%
speed) but never dies. Food fully restores its strength.

Multiple creatures (click the window to spawn one): they push each other
softly on contact (pure physics, no sensors, no rewards) and "conjugate" —
they exchange Q-table cells the way bacteria swap plasmids. Knowledge
flows to whoever does not have it yet.

## What to watch

A young creature darts around chaotically. After a few minutes of
simulation it starts climbing the smell gradient purposefully; the canvas
screen shows only the controls hint.
