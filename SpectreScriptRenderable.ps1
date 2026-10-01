# A Spectre renderable that rebuilds itself on every draw, so a live display follows the data
# instead of holding the first object it was given.
#
# It has to be dot-sourced *after* PwshSpectreConsole is imported: a class needs the Spectre.Console
# types to be loaded when the file is parsed, and PowerShell parses a script before running it.
# Which is also why this is a file of its own rather than a class inside the client.
#
#     Import-Module PwshSpectreConsole
#     . "$PSScriptRoot\SpectreScriptRenderable.ps1"
#     $live = [SpectreScriptRenderable]::new({ Get-MyTable })
#     Invoke-SpectreLive -Data $live -ScriptBlock { param($Context) $Context.refresh() }
#
# The base class, not IRenderable: Start-AnsiConsoleLive takes a Spectre.Console.Rendering.Renderable
# and refuses anything that merely implements the interface.
class SpectreScriptRenderable : Spectre.Console.Rendering.Renderable {
    # Called once per draw, so whatever it returns is rendered from scratch each time.
    [scriptblock] $Factory

    SpectreScriptRenderable([scriptblock] $factory) {
        $this.Factory = $factory
    }

    # Render is the one abstract member of the base class; Measure has a default implementation.
    [System.Collections.Generic.IEnumerable[Spectre.Console.Rendering.Segment]] Render([Spectre.Console.Rendering.RenderOptions] $options, [int] $maxWidth) {
        return (& $this.Factory).Render($options, $maxWidth)
    }
}
