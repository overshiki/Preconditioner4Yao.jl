using Yao
using Zygote
n_qubits = 3
hterm = put(n_qubits, 1=>Z)
hterm *= put(n_qubits, 2=>X)
hterm *= put(n_qubits, 3=>Y)
hterm += put(n_qubits, 1=>X)


struct Gate
    coeff::Number
    gtype::Symbol
    qubit::Int64
end


struct Circuit 
    expr::Expr 
    coeff::Number
end


function sym_grad(g::Gate, ::Val{:gx})
    return Gate(1im, :x, g.qubit)
end

function sym_grad(g::Gate, ::Val{:gy})
    return Gate(1im, :y, g.qubit)
end

function sym_grad(g::Gate, ::Val{:gz})
    return Gate(1im, :z, g.qubit)
end

function gate2expr(g::Gate, n_qubits::Int)
    g_dict = Dict(:gx=>:(Rx(0.1)),
                    :gy=>:(Ry(0.1)),
                    :gz=>:(Rz(0.1)),

                    :x=>:(X),
                    :y=>:(Y),
                    :z=>:(Z),
                )

    expr = :(put($n_qubits, $(g.qubit)=>$(g_dict[g.gtype])))
    return expr
end

function gates2circ(gates::Vector{Gate}, n_qubits::Int)
    chain = :(chain())
    for g in gates
        push!(chain.args, gate2expr(g, n_qubits))
    end
    return chain
end


function gates2coeff(gates::Vector{Gate})
    coeff = 1.0
    for g in gates 
        coeff *= g.coeff
    end
    return coeff
end


function get_grad_circuits(model_builder::Vector{Gate}, n_qubits::Int)
    grad_lead_circuits = Circuit[]
    for i in 1:length(model_builder)
        circ_left = model_builder[1:i-1]
        gate_center = model_builder[i]
        circ_right = model_builder[i+1:end]

        if gate_center.gtype in [:gx, :gy, :gz]
            circ_center = [sym_grad(gate_center, Val(gate_center.gtype)), gate_center]

            gates = Gate[]
            append!(gates, circ_left)
            append!(gates, circ_center)
            append!(gates, circ_right)
            circ = Circuit(gates2circ(gates, n_qubits), gates2coeff(gates))

            push!(grad_lead_circuits, circ)
        end        
    end
    return grad_lead_circuits
end



function prepare_circuit()
    model_builder = Gate[] 
    push!(model_builder, Gate(1.0, :gx, 1))
    push!(model_builder, Gate(1.0, :gy, 2))
    push!(model_builder, Gate(1.0, :gz, 1))
    return model_builder    
end

function prepare_hamiltonian()
    n_qubits = 3
    hterm = put(n_qubits, 1=>Z)
    hterm *= put(n_qubits, 2=>X)
    hterm *= put(n_qubits, 3=>Y)
    hterm += put(n_qubits, 1=>X)
    return hterm, n_qubits
end


function get_grad(hterm, circ_left, circ_right; input_left=randn(3), input_right=randn(3))

    function loss(pl, pr)
        cl = dispatch(circ_left, pl)
        cr = dispatch(circ_right, pr)
        regl = apply(zero_state(n_qubits), cl)
        regr = apply(apply(zero_state(n_qubits), cr), hterm)
        # @show statevec(regl)
        # @show state(regl)
        expectation = regl' * regr
        # return real(expectation)
        return imag(expectation)
    end

    # loss(randn(3), randn(3))
    grad = Zygote.gradient(loss, input_left, input_right)
    return grad

end



model_builder = prepare_circuit()
hterm, n_qubits = prepare_hamiltonian()


circ_right = gates2circ(model_builder, n_qubits) |> eval
grad_lead_circuits = get_grad_circuits(model_builder, n_qubits)

n_params = length(grad_lead_circuits)


grads = zeros(Float32, (n_params, n_params))
for (i, circ) in enumerate(grad_lead_circuits)
    # @show circ.expr, circ.coeff 
    circ_left = circ.expr |> eval 
    # only need grad on the right
    grad = get_grad(hterm, circ_left, circ_right; input_left=zeros(n_params), input_right=zeros(n_params))[1]
    # @show grad
    grads[i,:] .= grad

end

display(grads)

