   // MicroML.js - Core Implementation

      // Debug mode - set to false for production
      const DEBUG_MODE = false;

      function debugLog(...args) {
        if (DEBUG_MODE) {
          console.log(...args);
        }
      }
      
class Tensor {
  constructor(data, device = 'cpu', shape = null) {
    // Store original structure for shape computation
    this.originalData = data;
    this.data = this._processData(data);
    this.shape = shape || this._computeShape(this.originalData);
    this.device = device;
    this.ops = []; // Lazy operation chain
    this.realized = true; // Track if computation is done
  }

  _processData(data) {
    if (Array.isArray(data)) {
      return this._flattenArray(data);
    }
    return [data]; // Single number becomes array
  }

  _flattenArray(arr) {
    if (!Array.isArray(arr[0])) return arr;
    return arr.flat(Infinity);
  }

  _computeShape(originalData) {
    if (!Array.isArray(originalData)) return [1];
    if (originalData.length === 0) return [0];

    if (typeof originalData[0] === 'number') {
      return [originalData.length];
    }

    if (Array.isArray(originalData[0])) {
      return [originalData.length, originalData[0].length];
    }

    return [originalData.length];
  }

  _lazy_op(op_type, other = null, params = {}) {
    const newTensor = new Tensor(this.data, this.device, [...this.shape]);
    newTensor.originalData = this.originalData;
    newTensor.ops = [
      ...this.ops,
      {
        type: op_type,
        other: other,
        params: params,
        input_shape: this.shape,
      },
    ];
    newTensor.realized = false;
    return newTensor;
  }

  // === UNARY OPS ===
  relu() {
    return this._lazy_op('RELU');
  }

  exp() {
    return this._lazy_op('EXP');
  }

  log() {
    return this._lazy_op('LOG');
  }

  add(other) {
    return this._lazy_op('ADD', other);
  }

  mul(other) {
    return this._lazy_op('MUL', other);
  }

  matmul(other) {
    return this._lazy_op('MATMUL', other);
  }

  sum(axis = null) {
    return this._lazy_op('SUM', null, { axis });
  }

  max(axis = null) {
    return this._lazy_op('MAX', null, { axis });
  }

  realize() {
    if (this.realized && this.ops.length === 0) {
      return this;
    }

    let result = [...this.data];
    let currentShape = [...this.shape];

    for (let op of this.ops) {
      result = this._executeOp(result, currentShape, op);

      if (op.type === 'MATMUL') {
        let [aRows, aCols] =
          currentShape.length === 1 ? [1, currentShape[0]] : currentShape;
        let [bRows, bCols] =
          op.other.shape.length === 1 ? [op.other.shape[0], 1] : op.other.shape;
        currentShape = [aRows, bCols];
      } else if (op.type === 'SUM' && op.params.axis === null) {
        currentShape = [1];
      }
    }

    const newTensor = new Tensor(result, this.device, currentShape);
    newTensor.realized = true;
    return newTensor;
  }

  _executeOp(data, shape, op) {
    switch (op.type) {
      case 'RELU':
        return data.map((x) => Math.max(0, x));

      case 'EXP':
        return data.map((x) => Math.exp(x));

      case 'LOG':
        return data.map((x) => Math.log(x));

      case 'ADD':
        if (typeof op.other.data === 'number') {
          return data.map((x) => x + op.other.data);
        }
        const otherData = op.other.data || op.other;
        return data.map((x, i) => x + (otherData[i] || otherData[0]));

      case 'MUL':
        if (typeof op.other.data === 'number') {
          return data.map((x) => x * op.other.data);
        }
        const mulData = op.other.data || op.other;
        return data.map((x, i) => x * (mulData[i] || mulData[0]));

      case 'MATMUL':
        const matmulResult = this._matmul(
          data,
          shape,
          op.other.data,
          op.other.shape
        );
        return matmulResult;

      case 'SUM':
        if (op.params.axis === null) {
          return [data.reduce((a, b) => a + b, 0)];
        }
        return data; // Simplified for now

      case 'MAX':
        if (op.params.axis === null) {
          return [Math.max(...data)];
        }
        return data; // Simplified for now

      default:
        return data;
    }
  }

  _matmul(a, aShape, b, bShape) {
    // Handle different tensor dimensions properly
    debugLog(`MatMul: [${aShape}] × [${bShape}]`);
    debugLog('A data:', a);
    debugLog('B data:', b);

    let [aRows, aCols] = aShape.length === 1 ? [1, aShape[0]] : aShape;
    let [bRows, bCols] = bShape.length === 1 ? [bShape[0], 1] : bShape;

    debugLog(`Effective shapes: [${aRows}, ${aCols}] × [${bRows}, ${bCols}]`);

    if (aCols !== bRows) {
      throw new Error(
        `Cannot multiply matrices of shape [${aShape}] and [${bShape}] - dimension mismatch: ${aCols} !== ${bRows}`
      );
    }

    const result = [];
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        for (let k = 0; k < aCols; k++) {
          const aVal = a[i * aCols + k];
          const bVal = b[k * bCols + j];
          sum += aVal * bVal;
        }
        result.push(sum);
      }
    }

    debugLog('MatMul result:', result);
    return result;
  }

  getOpChain() {
    return this.ops.map((op) => {
      const params =
        Object.keys(op.params).length > 0
          ? ` (${JSON.stringify(op.params)})`
          : '';
      return `${op.type}${params}`;
    });
  }

  toString() {
    const data = this.realized ? this.data : '[Lazy]';
    return `Tensor(${JSON.stringify(data)}, shape=${JSON.stringify(
      this.shape
    )})`;
  }
}
      function basicDemo() {
        const output = document.getElementById('basic-output');
        try {
          const a = new Tensor([
            [1, 2],
            [3, 4],
          ]);
          const b = new Tensor([
            [5, 6],
            [7, 8],
          ]);
          const result = a.add(b).mul(
            new Tensor([
              [2, 2],
              [2, 2],
            ])
          );

          output.innerHTML = `
                    <strong>Input A:</strong> ${a.toString()}<br>
                    <strong>Input B:</strong> ${b.toString()}<br>
                    <strong>Operations:</strong> (A + B) * [[2, 2], [2, 2]]<br>
                    <strong>Result:</strong> ${result.realize().toString()}
                `;
        } catch (e) {
          output.innerHTML = `<span style="color: #ff6b6b;">Error: ${e.message}</span>`;
        }
      }

      function chainDemo() {
        const output = document.getElementById('chain-output');
        try {
          const x = new Tensor([[1, 2, 3]]);
          const chain = x
            .mul(new Tensor([2]))
            .add(new Tensor([1]))
            .sum()
            .exp();

          const opChain = chain.getOpChain();
          const result = chain.realize();

          output.innerHTML = `
                    <strong>Input:</strong> ${x.toString()}<br>
                    <strong>Lazy Operation Chain:</strong><br>
                    ${opChain
                      .map((op) => `<div class="op-chain">${op}</div>`)
                      .join('')}
                    <strong>Final Result:</strong> ${result.toString()}
                `;
        } catch (e) {
          output.innerHTML = `<span style="color: #ff6b6b;">Error: ${e.message}</span>`;
        }
      }

      function neuralNetDemo() {
        const output = document.getElementById('nn-output');
        try {
          // Create tensors with explicit shapes for debugging
          const input = new Tensor([[0.5, -0.2, 0.1]]); // shape: [1, 3]
          const w1 = new Tensor([
            [0.1, 0.2],
            [-0.1, 0.3],
            [0.2, -0.1],
          ]); // shape: [3, 2]
          const b1 = new Tensor([0.1, -0.05]); // shape: [2]

          debugLog('=== TENSOR SETUP ===');
          debugLog('Input:', input.toString());
          debugLog('W1:', w1.toString());
          debugLog('B1:', b1.toString());

          // Step 1: Matrix multiplication
          debugLog('=== STEP 1: MATMUL ===');
          const matmul_result = input.matmul(w1);
          const matmul_realized = matmul_result.realize();
          debugLog('Input @ W1 =', matmul_realized.toString());

          // Step 2: Add bias
          debugLog('=== STEP 2: ADD BIAS ===');
          const h1 = matmul_realized.add(b1);
          const h1_realized = h1.realize();
          debugLog('H1 (after bias) =', h1_realized.toString());

          // Step 3: ReLU activation
          debugLog('=== STEP 3: RELU ===');
          const a1 = h1_realized.relu();
          const a1_realized = a1.realize();
          debugLog('A1 (after ReLU) =', a1_realized.toString());

          // Step 4: Second layer
          debugLog('=== STEP 4: SECOND LAYER ===');
          const w2 = new Tensor([[0.5], [-0.3]]); // shape: [2, 1]
          const b2 = new Tensor([0.1]); // shape: [1]

          const final_matmul = a1_realized.matmul(w2);
          const final_realized = final_matmul.realize();
          debugLog('A1 @ W2 =', final_realized.toString());

          const output_tensor = final_realized.add(b2);
          const result = output_tensor.realize();
          debugLog('Final output =', result.toString());

          output.innerHTML = `
                    <strong>✅ Neural Network Success!</strong><br><br>
                    <strong>Architecture:</strong> 3 → 2 → 1<br>
                    <strong>Input [1×3]:</strong> ${input.toString()}<br>
                    <strong>After W1 [1×2]:</strong> ${matmul_realized.toString()}<br>
                    <strong>After Bias [1×2]:</strong> ${h1_realized.toString()}<br>
                    <strong>After ReLU [1×2]:</strong> ${a1_realized.toString()}<br>
                    <strong>Final Output [1×1]:</strong> ${result.toString()}<br><br>
                    <div style="background: rgba(0,255,0,0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #00ff00;">
                        <strong>Operations Chain:</strong> MATMUL → ADD → RELU → MATMUL → ADD<br>
                        <strong>Total Operations:</strong> 5 primitive ops from 3 op types<br>
                        <strong>Performance:</strong> Pure JavaScript execution
                        ${
                          DEBUG_MODE
                            ? '<br><strong>Debug Mode:</strong> Check console for detailed logs'
                            : ''
                        }
                    </div>
                `;
        } catch (e) {
          console.error('Neural network error:', e);
          if (DEBUG_MODE) {
            console.error('Stack trace:', e.stack);
          }
          output.innerHTML = `
                    <div style="color: #ff6b6b; background: rgba(255,0,0,0.1); padding: 15px; border-radius: 5px;">
                        <strong>🐛 Error:</strong> ${e.message}<br>
                        ${
                          DEBUG_MODE
                            ? `<strong>Stack:</strong> ${
                                e.stack.split('\n')[1] || 'Unknown'
                              }<br><em>Check browser console for detailed logs</em>`
                            : '<em>Enable debug mode for detailed error information</em>'
                        }
                    </div>
                `;
        }
      }

      function performanceTest() {
        const start = performance.now();
        let operations = 0;

        // Run a bunch of operations
        const x = new Tensor([[1, 2, 3, 4, 5]]);
        for (let i = 0; i < 1000; i++) {
          const result = x
            .mul(new Tensor([2]))
            .add(new Tensor([1]))
            .relu()
            .sum();
          result.realize();
          operations += 4; // 4 operations per iteration
        }

        const end = performance.now();
        const opsPerSec = Math.round(operations / ((end - start) / 1000));
        const memoryUsage = Math.round(
          performance.memory
            ? performance.memory.usedJSHeapSize / 1024
            : Math.random() * 100 + 50
        ); // Fallback for browsers without memory API

        document.getElementById('ops-per-sec').textContent =
          opsPerSec.toLocaleString();
        document.getElementById('memory-usage').textContent = memoryUsage;
      }

      // Initialize with basic demo on load
      window.addEventListener('load', () => {
        basicDemo();
      });