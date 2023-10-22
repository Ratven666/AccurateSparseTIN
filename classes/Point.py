from classes.abc_classes.PointABC import PointABC


class Point(PointABC):
    """
    Класс точки
    """

    __slots__ = []

    @classmethod
    def parse_point_from_db_row(cls, row: tuple):
        """
        Метод который создает и возвращает объект Point по данным читаемым из БД
        :param row: кортеж данных читаемых из БД
        :return: объект класса Point
        """
        return cls(id_=row[0], X=row[1], Y=row[2], Z=row[3], R=row[4], G=row[5], B=row[6])


if __name__ == "__main__":
    p1 = Point(1, 2, 3, 4, 5, 6, id_=13)
    p4 = Point(1, 2, 3, 4, 5, 6, id_=12)
    p5 = Point(1, 2, 3, 4, 5, 6)
    p2 = Point(10, 20, 30, 40, 50, 60)
    p3 = Point(10, 20, 30, 40, 50, 60)
    points = [p5]
    set_points = set()
    set_points.add(p1)
    # set_points.add(p2)
    # set_points.add(p3)
    # set_points.add(p4)
    set_points.add(p5)

    print(set_points)

    print(p1 in points)

