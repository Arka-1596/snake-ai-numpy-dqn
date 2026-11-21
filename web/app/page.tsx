
"use client";

import { useState, type CSSProperties } from "react";

const BOARD_SIZE = 12;

const SNAKE_PATH = new Set([
  27, 39, 51, 63, 75, 87,
  88, 89, 90, 78, 66, 54,
]);

const TILES = Array.from(
  { length: BOARD_SIZE * BOARD_SIZE },
  (_, index) => {
    const row = Math.floor(index / BOARD_SIZE);
    const column = index % BOARD_SIZE;

    return {
      index,
      dark: (row + column) % 2 === 0,
      snake: SNAKE_PATH.has(index),
      delay: `${(row + column) * 43}ms`,
    };
  }
);

export default function Home() {
  const [animationKey, setAnimationKey] = useState(0);

  function replayAnimation() {
    setAnimationKey((value) => value + 1);
  }

  return (
    <main className="site">
      <header className="masthead">
        <a className="brand" href="#" aria-label="Snake Studies home">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </span>
          <span>SNAKE / STUDIES</span>
        </a>

        <nav className="nav" aria-label="Main navigation">
          <a href="#study">THE STUDY</a>
          <a href="#method">THE METHOD</a>
          <a
            href="https://github.com/Arka-1596"
            target="_blank"
            rel="noopener noreferrer"
          >
            GITHUB ↗
          </a>
        </nav>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            AN EXPERIMENT IN LEARNING
            <span className="eyebrow-number">NO. 001 / 2026</span>
          </div>

          <h1 id="hero-title" className="hero-title">
            THE ART
            <span>OF</span>
            <em>LEARNING.</em>
          </h1>

          <div className="hero-lower">
            <p className="hero-description">
              A simple game.
              <br />
              Eleven observations.
              <br />
              A thousand decisions.
              <br />
              <strong>No instructions on how to win.</strong>
            </p>

            <a className="editorial-link" href="#study">
              EXPLORE THE STUDY
              <span aria-hidden="true">↘</span>
            </a>
          </div>

          <div className="hero-index">
            <span>ARTIFICIAL LEARNING, HUMAN CURIOSITY.</span>
            <span>01 — 04</span>
          </div>
        </div>

        <figure className="artwork">
          <div className="artwork-top">
            <span>FIG. 01 — THE MOVING PATTERN</span>
            <span>12 × 12</span>
          </div>

          <div className="checker-frame">
            <div
              key={animationKey}
              className="checkerboard"
              role="img"
              aria-label="Animated black and ivory checkerboard with a vermilion geometric snake"
            >
              {TILES.map((tile) => {
                const style = {
                  "--tile-delay": tile.delay,
                } as CSSProperties;

                return (
                  <span
                    key={tile.index}
                    className={[
                      "checker-tile",
                      tile.dark ? "checker-tile--dark" : "checker-tile--light",
                      tile.snake ? "checker-tile--snake" : "",
                    ].join(" ")}
                    style={style}
                  />
                );
              })}
            </div>

            <span className="frame-label frame-label--left">
              MOVEMENT / STUDY
            </span>
            <span className="frame-label frame-label--right">
              001
            </span>
          </div>

          <div className="artwork-bottom">
            <figcaption>
              A geometric interpretation of the Snake environment.
              <span>
                The live model will occupy this space in the next stage.
              </span>
            </figcaption>

            <button
              className="replay-button"
              onClick={replayAnimation}
              type="button"
            >
              <span aria-hidden="true">↻</span>
              REPLAY
            </button>
          </div>
        </figure>
      </section>

      <section className="stat-strip" id="study" aria-label="Project facts">
        <div className="stat">
          <span className="stat-number">11</span>
          <span className="stat-caption">OBSERVATIONS</span>
        </div>
        <div className="stat">
          <span className="stat-number">64</span>
          <span className="stat-caption">HIDDEN NEURONS</span>
        </div>
        <div className="stat">
          <span className="stat-number">03</span>
          <span className="stat-caption">POSSIBLE MOVES</span>
        </div>
        <div className="stat">
          <span className="stat-number">5.64</span>
          <span className="stat-caption">EVALUATION AVG.</span>
        </div>
      </section>

      <section className="manifesto" id="method">
        <div className="section-header">
          <span>02 / THE METHOD</span>
          <span>WHAT MAKES IT MOVE?</span>
        </div>

        <div className="manifesto-grid">
          <div className="manifesto-heading">
            <h2>
              A GAME
              <br />
              WITHOUT A
              <br />
              <em>PLAYBOOK.</em>
            </h2>
          </div>

          <div className="manifesto-content">
            <p>
              The snake was never told where to go. It was given a
              world to observe, a set of decisions to make, and
              consequences to learn from.
            </p>

            <div className="method-entry">
              <span>01 / OBSERVE</span>
              <p>
                Eleven numerical signals describe nearby danger,
                direction, and the position of food.
              </p>
            </div>

            <div className="method-entry">
              <span>02 / DECIDE</span>
              <p>
                A neural network built with NumPy estimates the
                value of moving straight, turning left, or
                turning right.
              </p>
            </div>

            <div className="method-entry">
              <span>03 / LEARN</span>
              <p>
                Rewards and experience replay allow the network
                to adjust its weights through backpropagation.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="closing">
        <div className="closing-marker">✳</div>

        <p className="closing-label">
          AN ONGOING EXPERIMENT
          <br />
          IN REINFORCEMENT LEARNING
        </p>

        <h2>
          TRIAL.
          <br />
          ERROR.
          <br />
          <em>EVOLUTION.</em>
        </h2>

        <div className="closing-bottom">
          <span>
            ORIGINAL IMPLEMENTATION
            <br />
            PYTHON / NUMPY / DQN
          </span>

          <a
            href="https://github.com/Arka-1596"
            target="_blank"
            rel="noopener noreferrer"
          >
            SOURCE CODE ↗
          </a>
        </div>
      </section>

      <footer className="footer">
        <span>© 2026 SNAKE STUDIES</span>
        <span>DESIGNED AS AN EXPERIMENT.</span>
        <span>END OF STUDY / FOR NOW.</span>
      </footer>
    </main>
  );
}
