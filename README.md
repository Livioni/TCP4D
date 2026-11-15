# TCP4D project page

Buildless English project page for TCP4D. The `page` branch is a standalone website and is published from its repository root. Keep the source repository private.

## Preview

Run `python3 -m http.server 8765` in this directory, then open `http://localhost:8765/`. All internal asset URLs are relative, including under GitHub Pages' `/TCP4D/` project path.

## Interactive demo

The Demo button in the hero links directly to
`https://huggingface.co/spaces/HarrisonPENG/TCP4D-DROID` in a new tab.

## Publishing

Push this branch to `origin/page`. In GitHub Pages settings, select **Deploy from a branch**, branch **page**, directory **/(root)**. `.nojekyll` enables direct static-file publication. The expected project URL is https://livioni.github.io/TCP4D/; the API-returned URL is authoritative.

GitHub requires an eligible account plan to publish Pages from a private repository. If the API rejects publication for this reason, keep the completed branch private and report the error. Do not make the repository public or upgrade the account automatically.

## Attribution

The layout is inspired by [Track4D](https://nirvanalan.github.io/projects/track4d/), with an original HTML/CSS/JavaScript implementation. Paper figures, branding and videos are the supplied TCP4D assets; their existing rights remain with their owners. Media remains hosted by its original source.
