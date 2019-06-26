using Documenter, DiscreteExteriorCalculus

makedocs(modules = [DiscreteExteriorCalculus],
    sitename="DiscreteExteriorCalculus.jl",
    pages = ["Main" => "index.md"])

deploydocs(
    repo = "github.com/rigetti/DiscreteExteriorCalculus.jl.git",
)
