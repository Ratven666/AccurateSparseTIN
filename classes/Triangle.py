from classes.Point import Point


class Triangle:

    __slots__ = ["id", "point_0", "point_1", "point_2", "r", "mse", "vv", "container_dict"]

    def __init__(self, point_0: Point, point_1: Point, point_2: Point, r=None, mse=None, id_=None):
        self.id = id_
        self.point_0 = point_0
        self.point_1 = point_1
        self.point_2 = point_2
        self.r = r
        self.mse = mse
        self.vv = 0
        self.container_dict = {}

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id}\t[[Point_0: [id: {self.point_0.id},\t" \
               f"x: {self.point_0.X} y: {self.point_0.Y} z: {self.point_0.Z}]\t" \
               f"\t\t [Point_1: [id: {self.point_1.id},\t" \
               f"x: {self.point_1.X} y: {self.point_1.Y} z: {self.point_1.Z}]\t" \
               f"\t\t [Point_2: [id: {self.point_2.id},\t" \
               f"x: {self.point_2.X} y: {self.point_2.Y} z: {self.point_2.Z}]\t" \
               f"r: {self.r},\tmse: {self.mse}"

    def __repr__(self):
        return f"{self.__class__.__name__} [id={self.id}, points=[{self.point_0.id}-" \
               f"{self.point_1.id}-{self.point_2.id}]]"

    def __iter__(self):
        point_lst = [self.point_0, self.point_1, self.point_2]
        return iter(point_lst)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            raise TypeError("Операнд справа должен иметь тип Triangle")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.point_0 == other.point_0) and \
                   (self.point_1 == other.point_1) and \
                   (self.point_2 == other.point_2)
        return False

    def get_z_from_xy(self, x, y):
        """
        Рассчитывает отметку точки (x, y) в плоскости треугольника
        :param x: координата x
        :param y: координата y
        :return: координата z для точки (x, y)
        """
        a = -((self.point_1.Y - self.point_0.Y) * (self.point_2.Z - self.point_0.Z) -
              (self.point_2.Y - self.point_0.Y) * (self.point_1.Z - self.point_0.Z))
        b = ((self.point_1.X - self.point_0.X) * (self.point_2.Z - self.point_0.Z) -
             (self.point_2.X - self.point_0.X) * (self.point_1.Z - self.point_0.Z))
        c = -((self.point_1.X - self.point_0.X) * (self.point_2.Y - self.point_0.Y) -
              (self.point_2.X - self.point_0.X) * (self.point_1.Y - self.point_0.Y))
        d = -(self.point_0.X * a + self.point_0.Y * b + self.point_0.Z * c)
        try:
            z = (a * x + b * y + d) / -c
        except ZeroDivisionError:
            return None
        return z

    def get_area(self):
        """
        Рассчитывает площадь проекции треугольника на горизонтальной плоскости
        """
        a = ((self.point_1.X - self.point_0.X)**2 + (self.point_1.Y - self.point_0.Y)**2) ** 0.5
        b = ((self.point_2.X - self.point_1.X)**2 + (self.point_2.Y - self.point_1.Y)**2) ** 0.5
        c = ((self.point_0.X - self.point_2.X)**2 + (self.point_0.Y - self.point_2.Y)**2) ** 0.5
        p = (a + b + c) / 2
        s = (p * (p - a) * (p - b) * (p - c)) ** 0.5
        return s

    def is_point_in_triangle(self, point: Point):
        """
        Проверяет попадает ли точка внутрь треугольника по критерию суммы площадей
        :param point: точка для которой выполняется проверка
        :return: True - если точка внутри треугольника и False если нет
        """
        s_abc = self.get_area()
        if s_abc == 0:
            return False
        s_ab_p = Triangle(self.point_0, self.point_1, point).get_area()
        s_bc_p = Triangle(self.point_1, self.point_2, point).get_area()
        s_ca_p = Triangle(self.point_2, self.point_0, point).get_area()
        delta_s = abs(s_abc - (s_ab_p + s_bc_p + s_ca_p))
        if delta_s < 1e-6:
            return True
        return False

    @classmethod
    def parse_triangle_from_db_row(cls, row: tuple):
        """
        Метод который создает и возвращает объект Triangle по данным читаемым из БД
        :param row: кортеж данных читаемых из БД
        :return: объект класса Triangle
        """
        id_ = row[0]
        r = row[1]
        mse = row[2]
        point_0 = Point.parse_point_from_db_row(row[3:10])
        point_1 = Point.parse_point_from_db_row(row[10:17])
        point_2 = Point.parse_point_from_db_row(row[17:])
        return cls(id_=id_, r=r, mse=mse, point_0=point_0, point_1=point_1, point_2=point_2)

    def get_dict(self):
        """
        Возвращает словарь с данными треугольника
        """
        return {"id": self.id,
                "r": self.r,
                "mse": self.mse,
                "point_0": self.point_0.get_dict(),
                "point_1": self.point_1.get_dict(),
                "point_2": self.point_2.get_dict()}


if __name__ == "__main__":
    p0 = Point(0, 0, 1, 0, 0, 0)
    p1 = Point(10, 0, 0, 0, 0, 0)
    p2 = Point(0, 10, 28.01, 0, 0, 0)
    p3 = Point(20, 0, 0, 0, 0, 0)
    p4 = Point(5, 5, 5, 0, 0, 0)
    p = Point(1, -0.001, 5, 0, 0, 0)
    tri = Triangle(p0, p1, p2)
    print(tri.get_z_from_xy(1, 1))
    print(tri.is_point_in_triangle(p))
    for idx, point in enumerate(tri):
        print(idx, point)
