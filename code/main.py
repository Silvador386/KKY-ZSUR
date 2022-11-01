import load
import plot

DATA_PATH = "../data/data.txt"


def main():
    data = load.load_data(data_path=DATA_PATH)

    plot.plot_2D(data)



if __name__ == "__main__":
    main()
