/**
 * AI Maze Solver — Q-Learning Visualization
 * Main application: maze generation, Q-Learning engine, canvas rendering
 */

// ==================== MAZE GENERATION (DFS) ====================
class Maze {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.walls = new Set();
        this.start = [0, 0];
        this.goal = [height - 1, width - 1];
        this.generate();
    }

    _key(c1, c2) {
        const a = c1[0] < c2[0] || (c1[0] === c2[0] && c1[1] < c2[1]) ? c1 : c2;
        const b = a === c1 ? c2 : c1;
        return `${a[0]},${a[1]}|${b[0]},${b[1]}`;
    }

    generate() {
        this.walls = new Set();
        const { width: w, height: h } = this;
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                if (r > 0) this.walls.add(this._key([r, c], [r - 1, c]));
                if (c > 0) this.walls.add(this._key([r, c], [r, c - 1]));
            }
        }
        const visited = new Set();
        const stack = [[0, 0]];
        visited.add('0,0');
        while (stack.length) {
            const [r, c] = stack[stack.length - 1];
            const neighbors = [];
            for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                const nr = r+dr, nc = c+dc;
                if (nr>=0 && nr<h && nc>=0 && nc<w && !visited.has(`${nr},${nc}`))
                    neighbors.push([nr, nc]);
            }
            if (neighbors.length) {
                const next = neighbors[Math.floor(Math.random()*neighbors.length)];
                this.walls.delete(this._key([r,c], next));
                visited.add(`${next[0]},${next[1]}`);
                stack.push(next);
            } else {
                stack.pop();
            }
        }
    }

    canMove(from, to) {
        const [r, c] = to;
        if (r<0||r>=this.height||c<0||c>=this.width) return false;
        return !this.walls.has(this._key(from, to));
    }

    toJSON() {
        return { width: this.width, height: this.height, walls: [...this.walls], start: this.start, goal: this.goal };
    }

    static fromPython(data) {
        const m = new Maze(data.width, data.height);
        m.walls = new Set();
        for (const w of data.walls) {
            m.walls.add(m._key(w[0], w[1]));
        }
        m.start = data.start;
        m.goal = data.goal;
        return m;
    }
}

// ==================== Q-LEARNING ENGINE ====================
class QLearning {
    constructor(stateSize, actionSize, lr=0.1, gamma=0.99, epsStart=1.0, epsEnd=0.01, epsDecay=0.995) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.lr = lr;
        this.gamma = gamma;
        this.epsilon = epsStart;
        this.epsEnd = epsEnd;
        this.epsDecay = epsDecay;
        this.qTable = {};
    }

    getQ(s) {
        if (!this.qTable[s]) this.qTable[s] = new Array(this.actionSize).fill(0);
        return this.qTable[s];
    }

    selectAction(state, training=true) {
        if (training && Math.random() < this.epsilon)
            return Math.floor(Math.random() * this.actionSize);
        const q = this.getQ(state);
        let maxQ = -Infinity, best = 0;
        for (let a = 0; a < q.length; a++) { if (q[a] > maxQ) { maxQ = q[a]; best = a; } }
        return best;
    }

    update(s, a, r, ns, done) {
        const q = this.getQ(s);
        const nq = this.getQ(ns);
        const target = done ? r : r + this.gamma * Math.max(...nq);
        q[a] += this.lr * (target - q[a]);
    }

    decayEpsilon() {
        this.epsilon = Math.max(this.epsEnd, this.epsilon * this.epsDecay);
    }

    loadFromJSON(data) {
        this.qTable = {};
        for (const [k, v] of Object.entries(data.q_table)) this.qTable[parseInt(k)] = [...v];
        this.epsilon = data.epsilon;
    }
}

// ==================== MAZE ENVIRONMENT (JS) ====================
class MazeEnvJS {
    constructor(maze) {
        this.maze = maze;
        this.deltas = [[-1,0],[0,1],[1,0],[0,-1]]; // UP RIGHT DOWN LEFT
        this.maxSteps = maze.width * maze.height * 4;
        this.reset();
    }

    posToState(p) { return p[0] * this.maze.width + p[1]; }

    reset() {
        this.agentPos = [...this.maze.start];
        this.steps = 0;
        this.visited = new Set();
        this.visited.add(`${this.agentPos[0]},${this.agentPos[1]}`);
        this.path = [[...this.agentPos]];
        return this.posToState(this.agentPos);
    }

    step(action) {
        const [dr, dc] = this.deltas[action];
        const newPos = [this.agentPos[0]+dr, this.agentPos[1]+dc];
        this.steps++;
        let reward, hitWall = false;

        if (this.maze.canMove(this.agentPos, newPos)) {
            this.agentPos = newPos;
            this.visited.add(`${newPos[0]},${newPos[1]}`);
            this.path.push([...newPos]);
        } else {
            hitWall = true;
        }

        const atGoal = this.agentPos[0]===this.maze.goal[0] && this.agentPos[1]===this.maze.goal[1];
        if (atGoal) reward = 100;
        else if (hitWall) reward = -5;
        else reward = -1;

        const terminated = atGoal;
        const truncated = this.steps >= this.maxSteps;
        return { state: this.posToState(this.agentPos), reward, terminated, truncated };
    }
}

// ==================== RENDERER ====================
class MazeRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    render(maze, agentPos, visited, solutionPath, qAgent) {
        const ctx = this.ctx;
        const W = this.canvas.width, H = this.canvas.height;
        const cellW = W / maze.width, cellH = H / maze.height;

        ctx.fillStyle = '#0d1225';
        ctx.fillRect(0, 0, W, H);

        // Visited cells
        if (visited) {
            for (const key of visited) {
                const [r, c] = key.split(',').map(Number);
                ctx.fillStyle = 'rgba(99,102,241,0.08)';
                ctx.fillRect(c*cellW+1, r*cellH+1, cellW-2, cellH-2);
            }
        }

        // Solution path
        if (solutionPath && solutionPath.length > 1) {
            ctx.strokeStyle = 'rgba(6,214,160,0.4)';
            ctx.lineWidth = Math.max(2, cellW * 0.15);
            ctx.lineCap = 'round'; ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(solutionPath[0][1]*cellW+cellW/2, solutionPath[0][0]*cellH+cellH/2);
            for (let i=1; i<solutionPath.length; i++)
                ctx.lineTo(solutionPath[i][1]*cellW+cellW/2, solutionPath[i][0]*cellH+cellH/2);
            ctx.stroke();
        }

        // Q-value arrows (subtle)
        if (qAgent && Object.keys(qAgent.qTable).length > 0) {
            const arrowLen = Math.min(cellW, cellH) * 0.25;
            for (let r = 0; r < maze.height; r++) {
                for (let c = 0; c < maze.width; c++) {
                    const s = r * maze.width + c;
                    const q = qAgent.qTable[s];
                    if (!q) continue;
                    const maxQ = Math.max(...q);
                    if (maxQ === 0) continue;
                    const bestA = q.indexOf(maxQ);
                    const cx = c*cellW+cellW/2, cy = r*cellH+cellH/2;
                    const dirs = [[0,-1],[1,0],[0,1],[-1,0]]; // UP RIGHT DOWN LEFT
                    const [dx, dy] = dirs[bestA];
                    ctx.strokeStyle = 'rgba(99,102,241,0.2)';
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(cx+dx*arrowLen, cy+dy*arrowLen);
                    ctx.stroke();
                }
            }
        }

        // Walls
        ctx.strokeStyle = '#2d3352';
        ctx.lineWidth = 2;
        // Borders
        ctx.strokeRect(0, 0, W, H);
        // Internal walls
        for (const wallKey of maze.walls) {
            const [p1, p2] = wallKey.split('|').map(s => s.split(',').map(Number));
            const [r1,c1] = p1, [r2,c2] = p2;
            if (r1 === r2) { // horizontal neighbors -> vertical wall
                const wc = Math.max(c1, c2);
                ctx.beginPath();
                ctx.moveTo(wc*cellW, r1*cellH);
                ctx.lineTo(wc*cellW, r1*cellH+cellH);
                ctx.stroke();
            } else { // vertical neighbors -> horizontal wall
                const wr = Math.max(r1, r2);
                ctx.beginPath();
                ctx.moveTo(c1*cellW, wr*cellH);
                ctx.lineTo(c1*cellW+cellW, wr*cellH);
                ctx.stroke();
            }
        }

        // Start cell
        const [sr, sc] = maze.start;
        const pad = cellW * 0.15;
        ctx.fillStyle = '#06d6a0';
        ctx.shadowColor = 'rgba(6,214,160,0.6)'; ctx.shadowBlur = 12;
        ctx.fillRect(sc*cellW+pad, sr*cellH+pad, cellW-pad*2, cellH-pad*2);
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.max(10, cellW*0.35)}px Outfit`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('S', sc*cellW+cellW/2, sr*cellH+cellH/2);

        // Goal cell
        const [gr, gc] = maze.goal;
        ctx.fillStyle = '#f72585';
        ctx.shadowColor = 'rgba(247,37,133,0.6)'; ctx.shadowBlur = 12;
        ctx.fillRect(gc*cellW+pad, gr*cellH+pad, cellW-pad*2, cellH-pad*2);
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fff';
        ctx.fillText('G', gc*cellW+cellW/2, gr*cellH+cellH/2);

        // Agent
        if (agentPos) {
            const [ar, ac] = agentPos;
            const cx = ac*cellW+cellW/2, cy = ar*cellH+cellH/2;
            const radius = Math.min(cellW, cellH) * 0.3;
            // Glow
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius*2.5);
            grad.addColorStop(0, 'rgba(99,102,241,0.4)');
            grad.addColorStop(1, 'rgba(99,102,241,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(cx-radius*2.5, cy-radius*2.5, radius*5, radius*5);
            // Body
            ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI*2);
            ctx.fillStyle = '#6366f1';
            ctx.shadowColor = 'rgba(99,102,241,0.8)'; ctx.shadowBlur = 15;
            ctx.fill(); ctx.shadowBlur = 0;
            // Eyes
            ctx.fillStyle = '#fff';
            ctx.beginPath(); ctx.arc(cx-radius*0.25, cy-radius*0.15, radius*0.15, 0, Math.PI*2); ctx.fill();
            ctx.beginPath(); ctx.arc(cx+radius*0.25, cy-radius*0.15, radius*0.15, 0, Math.PI*2); ctx.fill();
        }
    }
}

// ==================== CHART ====================
class MiniChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = [];
    }

    push(value) {
        this.data.push(value);
        if (this.data.length > 200) this.data.shift();
    }

    clear() { this.data = []; }

    render() {
        const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
        ctx.fillStyle = '#0d1225'; ctx.fillRect(0, 0, W, H);
        if (this.data.length < 2) return;

        // Smooth with moving average
        const window = Math.max(1, Math.floor(this.data.length / 50));
        const smooth = [];
        for (let i = 0; i < this.data.length; i++) {
            let sum = 0, count = 0;
            for (let j = Math.max(0, i-window); j <= i; j++) { sum += this.data[j]; count++; }
            smooth.push(sum/count);
        }

        const min = Math.min(...smooth), max = Math.max(...smooth);
        const range = max - min || 1;
        const padX = 5, padY = 10;

        // Gradient fill
        const grad = ctx.createLinearGradient(0, padY, 0, H-padY);
        grad.addColorStop(0, 'rgba(99,102,241,0.3)');
        grad.addColorStop(1, 'rgba(99,102,241,0)');

        ctx.beginPath();
        ctx.moveTo(padX, H - padY);
        for (let i = 0; i < smooth.length; i++) {
            const x = padX + (i / (smooth.length-1)) * (W - padX*2);
            const y = H - padY - ((smooth[i]-min)/range) * (H - padY*2);
            ctx.lineTo(x, y);
        }
        ctx.lineTo(padX + (W-padX*2), H-padY);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Line
        ctx.beginPath();
        for (let i = 0; i < smooth.length; i++) {
            const x = padX + (i / (smooth.length-1)) * (W - padX*2);
            const y = H - padY - ((smooth[i]-min)/range) * (H - padY*2);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2; ctx.stroke();

        // Label
        ctx.fillStyle = '#5c6478'; ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'left'; ctx.fillText('Reward', padX, 12);
    }
}

// ==================== APP ====================
class App {
    constructor() {
        this.maze = null;
        this.env = null;
        this.agent = null;
        this.renderer = new MazeRenderer(document.getElementById('maze-canvas'));
        this.chart = new MiniChart(document.getElementById('chart-canvas'));

        this.isTraining = false;
        this.isSolving = false;
        this.animSpeed = 50;
        this.trainAnimId = null;
        this.solveAnimId = null;

        this.history = { rewards: [], steps: [], solved: [] };

        this._bindUI();
        this._resizeCanvas();
        window.addEventListener('resize', () => this._resizeCanvas());
    }

    _resizeCanvas() {
        const wrapper = document.getElementById('canvas-wrapper');
        const size = Math.min(wrapper.clientWidth - 32, wrapper.clientHeight - 32, 600);
        const canvas = document.getElementById('maze-canvas');
        canvas.width = size; canvas.height = size;
        if (this.maze) this._draw();
    }

    _bindUI() {
        // Maze size slider
        const sizeSlider = document.getElementById('maze-size');
        const sizeVal = document.getElementById('maze-size-value');
        sizeSlider.oninput = () => sizeVal.textContent = `${sizeSlider.value} × ${sizeSlider.value}`;

        // Episodes slider
        const epSlider = document.getElementById('train-episodes');
        const epVal = document.getElementById('train-episodes-value');
        epSlider.oninput = () => epVal.textContent = epSlider.value;

        // Learning rate slider
        const lrSlider = document.getElementById('learning-rate');
        const lrVal = document.getElementById('learning-rate-value');
        lrSlider.oninput = () => lrVal.textContent = parseFloat(lrSlider.value).toFixed(2);

        // Discount slider
        const gSlider = document.getElementById('discount-factor');
        const gVal = document.getElementById('discount-factor-value');
        gSlider.oninput = () => gVal.textContent = parseFloat(gSlider.value).toFixed(2);

        // Speed slider
        const speedSlider = document.getElementById('animation-speed');
        const speedVal = document.getElementById('animation-speed-value');
        speedSlider.oninput = () => { this.animSpeed = parseInt(speedSlider.value); speedVal.textContent = speedSlider.value; };

        // Buttons
        document.getElementById('btn-generate').onclick = () => this.generateMaze();
        document.getElementById('btn-load-python').onclick = () => this.loadFromPython();
        document.getElementById('btn-train').onclick = () => this.startTraining();
        document.getElementById('btn-stop').onclick = () => this.stopAll();
        document.getElementById('btn-solve').onclick = () => this.solveMaze();
        document.getElementById('btn-reset-agent').onclick = () => this.resetAgent();
    }

    generateMaze() {
        const size = parseInt(document.getElementById('maze-size').value);
        this.maze = new Maze(size, size);
        this.env = new MazeEnvJS(this.maze);
        this.agent = new QLearning(size*size, 4,
            parseFloat(document.getElementById('learning-rate').value),
            parseFloat(document.getElementById('discount-factor').value)
        );
        this.history = { rewards: [], steps: [], solved: [] };
        this.chart.clear();

        document.getElementById('maze-size-label').textContent = `${size} × ${size}`;
        document.getElementById('canvas-overlay').classList.add('hidden');
        document.getElementById('btn-train').disabled = false;
        document.getElementById('btn-solve').disabled = false;
        this._updateStats(0, 0, 0, 1.0, 0, 0);
        this._draw();
    }

    async loadFromPython() {
        try {
            const [mazeRes, qRes, histRes] = await Promise.all([
                fetch('maze_layout.json'), fetch('maze_qtable.json'), fetch('training_history.json').catch(()=>null)
            ]);
            const mazeData = await mazeRes.json();
            const qData = await qRes.json();

            this.maze = Maze.fromPython(mazeData);
            this.env = new MazeEnvJS(this.maze);
            this.agent = new QLearning(mazeData.width*mazeData.height, 4);
            this.agent.loadFromJSON(qData);

            if (histRes && histRes.ok) {
                const hData = await histRes.json();
                this.history.rewards = hData.episode_rewards || [];
                hData.episode_rewards?.forEach(r => this.chart.push(r));
            }

            document.getElementById('maze-size').value = mazeData.width;
            document.getElementById('maze-size-value').textContent = `${mazeData.width} × ${mazeData.height}`;
            document.getElementById('maze-size-label').textContent = `${mazeData.width} × ${mazeData.height}`;
            document.getElementById('canvas-overlay').classList.add('hidden');
            document.getElementById('btn-train').disabled = false;
            document.getElementById('btn-solve').disabled = false;

            this._updateStats(this.history.rewards.length, 0, 0, this.agent.epsilon,
                0, Object.keys(this.agent.qTable).length);
            this._draw();
            this.chart.render();
            this._setStatus('Loaded', '');
        } catch (e) {
            alert('Could not load Python data. Make sure maze_layout.json and maze_qtable.json exist in the web/ folder.\n\nRun: python train_maze.py');
        }
    }

    startTraining() {
        if (!this.maze || this.isTraining) return;
        this.isTraining = true;
        this._setStatus('Training...', 'training');
        document.getElementById('btn-train').disabled = true;
        document.getElementById('btn-stop').disabled = false;
        document.getElementById('btn-solve').disabled = true;

        const totalEps = parseInt(document.getElementById('train-episodes').value);
        let ep = 0;

        const runEpisode = () => {
            if (!this.isTraining || ep >= totalEps) {
                this.isTraining = false;
                this._setStatus('Trained', '');
                document.getElementById('btn-train').disabled = false;
                document.getElementById('btn-stop').disabled = true;
                document.getElementById('btn-solve').disabled = false;
                this._draw();
                return;
            }

            let state = this.env.reset();
            let totalReward = 0, steps = 0, done = false;

            while (!done && steps < this.maze.width * this.maze.height * 4) {
                const action = this.agent.selectAction(state, true);
                const result = this.env.step(action);
                this.agent.update(state, action, result.reward, result.state, result.terminated);
                state = result.state;
                totalReward += result.reward;
                steps++;
                done = result.terminated || result.truncated;
            }
            this.agent.decayEpsilon();
            ep++;

            this.history.rewards.push(totalReward);
            this.history.steps.push(steps);
            this.history.solved.push(this.env.agentPos[0]===this.maze.goal[0] && this.env.agentPos[1]===this.maze.goal[1]);
            this.chart.push(totalReward);

            const recent = this.history.solved.slice(-50);
            const solveRate = recent.length ? Math.round(recent.filter(Boolean).length/recent.length*100) : 0;
            this._updateStats(ep, totalReward, steps, this.agent.epsilon, solveRate, Object.keys(this.agent.qTable).length);

            // Draw occasionally
            if (ep % Math.max(1, Math.floor(101-this.animSpeed)) === 0 || ep === totalEps) {
                this._draw();
                this.chart.render();
            }

            const delay = Math.max(0, 100 - this.animSpeed);
            this.trainAnimId = setTimeout(runEpisode, delay);
        };

        runEpisode();
    }

    solveMaze() {
        if (!this.maze || !this.agent || this.isSolving) return;
        this.isSolving = true;
        this._setStatus('Solving...', 'solving');
        document.getElementById('btn-solve').disabled = true;
        document.getElementById('btn-stop').disabled = false;

        let state = this.env.reset();
        this._draw();

        const stepFn = () => {
            if (!this.isSolving) return;
            const action = this.agent.selectAction(state, false);
            const result = this.env.step(action);
            state = result.state;
            this._draw(this.env.path);

            if (result.terminated) {
                this.isSolving = false;
                this._setStatus('Solved!', '');
                document.getElementById('btn-solve').disabled = false;
                document.getElementById('btn-stop').disabled = true;
                return;
            }
            if (result.truncated) {
                this.isSolving = false;
                this._setStatus('Failed', '');
                document.getElementById('btn-solve').disabled = false;
                document.getElementById('btn-stop').disabled = true;
                return;
            }
            const delay = Math.max(10, 200 - this.animSpeed * 2);
            this.solveAnimId = setTimeout(stepFn, delay);
        };
        stepFn();
    }

    stopAll() {
        this.isTraining = false;
        this.isSolving = false;
        if (this.trainAnimId) clearTimeout(this.trainAnimId);
        if (this.solveAnimId) clearTimeout(this.solveAnimId);
        this._setStatus('Stopped', '');
        document.getElementById('btn-train').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        document.getElementById('btn-solve').disabled = false;
    }

    resetAgent() {
        if (this.maze) {
            this.agent = new QLearning(this.maze.width*this.maze.height, 4,
                parseFloat(document.getElementById('learning-rate').value),
                parseFloat(document.getElementById('discount-factor').value)
            );
            this.env.reset();
            this.history = { rewards: [], steps: [], solved: [] };
            this.chart.clear(); this.chart.render();
            this._updateStats(0, 0, 0, 1.0, 0, 0);
            this._draw();
            this._setStatus('Reset', '');
        }
    }

    _draw(solutionPath = null) {
        if (!this.maze) return;
        this.renderer.render(this.maze, this.env?.agentPos, this.env?.visited, solutionPath, this.agent);
    }

    _updateStats(ep, reward, steps, eps, solveRate, qEntries) {
        document.getElementById('stat-episode').textContent = ep;
        document.getElementById('stat-reward').textContent = typeof reward === 'number' ? reward.toFixed(0) : reward;
        document.getElementById('stat-steps').textContent = steps;
        document.getElementById('stat-epsilon').textContent = eps.toFixed(3);
        document.getElementById('stat-solve-rate').textContent = `${solveRate}%`;
        document.getElementById('stat-q-entries').textContent = qEntries;
    }

    _setStatus(text, className) {
        const badge = document.getElementById('badge-status');
        badge.textContent = text;
        badge.className = 'badge badge-status ' + className;
    }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => new App());
