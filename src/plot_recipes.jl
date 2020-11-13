using RecipesBase
using ColorTypes
using LaTeXStrings
import PlotUtils: cgrad


const mygreen = RGBA{Float64}(151/255,180/255,118/255,1)
const mygreen2 = RGBA{Float64}(113/255,161/255,103/255,1)
const myblue = RGBA{Float64}(74/255,144/255,226/255,1)


## Main scalar grid data plot recipe

@recipe function f(x::AbstractRange,y::AbstractRange,w::T;trim=0) where {T<:ScalarGridData}
    seriestype --> :contour
    aspect_ratio := 1
    legend --> :none
    grid --> :none
    linewidth --> 1
    framestyle --> :frame
    xlims --> (-Inf,Inf)
    ylims --> (-Inf,Inf)
    levels --> range(minimum(w.data),stop=maximum(w.data),length=16)
    x[1+trim:end-trim],y[1+trim:end-trim],transpose(w.data[1+trim:end-trim,1+trim:end-trim])
end

@recipe function f(w::T,g::PhysicalGrid) where {T<:ScalarGridData}
    xg, yg = coordinates(w,g)
    xg, yg, w
end

@recipe function f(w::T) where {T<:ScalarGridData}
    m,n = size(w.data)
    1:m, 1:n, w
end


## Vector grid data

@recipe function f(xu::AbstractRange,yu::AbstractRange,xv::AbstractRange,yv::AbstractRange,q::T) where {T<:VectorGridData}
    layout := (1,2)
    @series begin
      subplot := 1
      title --> L"u"
      xu, yu, q.u
    end

    @series begin
      subplot := 2
      title --> L"v"
      xv, yv, q.v
    end

end

@recipe function f(q::T,g::PhysicalGrid) where {T<:VectorGridData}
    xu, yu, xv, yv = coordinates(q,g)
    return xu, yu, xv, yv, q
end

@recipe function f(q::T) where {T<:VectorGridData}
    mu,nu = size(q.u)
    mv,nv = size(q.v)
    return 1:mu, 1:nu, 1:mv, 1:nv, q
end

## Tensor grid data

@recipe function f(xdudx::AbstractRange,ydudx::AbstractRange,
                   xdudy::AbstractRange,ydudy::AbstractRange,
                   xdvdx::AbstractRange,ydvdx::AbstractRange,
                   xdvdy::AbstractRange,ydvdy::AbstractRange,q::T) where {T<:TensorGridData}
    layout := (2,2)
    @series begin
      subplot := 1
      title --> L"du/dx"
      xdudx, ydudx, q.dudx
    end

    @series begin
      subplot := 2
      title --> L"dv/dx"
      xdvdx, ydvdx, q.dvdx
    end

    @series begin
      subplot := 3
      title --> L"du/dy"
      xdudy, ydudy, q.dudy
    end

    @series begin
      subplot := 4
      title --> L"dv/dy"
      xdvdy, ydvdy, q.dvdy
    end

end

@recipe function f(q::T,g::PhysicalGrid) where {T<:TensorGridData}
    xdudx, ydudx = coordinates(q.dudx,g)
    xdudy, ydudy = coordinates(q.dudy,g)
    xdvdx, ydvdx = coordinates(q.dvdx,g)
    xdvdy, ydvdy = coordinates(q.dvdy,g)
    return xdudx, ydudx, xdudy, ydudy, xdvdx, ydvdx, xdvdy, ydvdy, q
end

@recipe function f(q::T) where {T<:TensorGridData}
    mdudx,ndudx = size(q.dudx)
    mdudy,ndudy = size(q.dudy)
    mdvdx,ndvdx = size(q.dvdx)
    mdvdy,ndvdy = size(q.dvdy)
    return 1:mdudx, 1:ndudx, 1:mdudy, 1:ndudy, 1:mdvdx, 1:ndvdx, 1:mdvdy, 1:ndvdy, q
end

## Generated fields

@recipe function f(field::GeneratedField)
    field(), field.grid
end
