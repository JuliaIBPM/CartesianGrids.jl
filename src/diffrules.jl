using DiffRules

DiffRules.@define_diffrule CartesianGrids.Regularize(x,y,dx) = :(1), :(1)

